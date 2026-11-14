import functools, jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from . import _utils

#   Readme.md
#     This file contains one function meant to be called from outside this file, fused_grad (_kernel and _epilogue are internal). This function takes in
#     full matrices (2d jax arrays) of size ~ H/V/BT and updates the weight and state matrices exactly like its discussed in the FORGE research paper.
#     The other one, optimizer_only (commented at the end of the file), assumes that the dLoss/dw matrix is fully materialized and passed. nothing uses it yet, see its comment.




















#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



@functools.partial(jax.jit, static_argnames = ("optimizer", "state_mode", "interpret", "tiles"))
# we want it to recompile for different optimizers, state_modes, interpret modes and tile sizes
def  fused_grad(go, x, w, states, scalars, *, optimizer, state_mode, interpret, tiles = None):
    #    go is (BT, V)    #    x is (BT, H)    #    w is (V, H)    #    are all jax.Arrays (2D) and so is scalars (1D)
    # optimizer and state_modes are strings, states is a dict(string : dict(jax.Array (rows, cols) dtype))
    opt, codec = _utils.lookup(optimizer, state_mode)                                                                                 # returns classes for strings
    (BT, V), H = go.shape, x.shape[1]                                                                                                 #
    bv, bh = _utils.pick_tiles(V, H, tiles, codec)                                                                                    # validates and returns grid size, default if None
    bbt = _utils.pick_bt_tile(BT, tiles)                                                                                              #
    return pl.pallas_call(                                                                                                            #
        functools.partial(_kernel, opt = opt, codec = codec, BT = BT),                                                                #
        grid = (pl.cdiv(V, bv), pl.cdiv(H, bh), pl.cdiv(BT, bbt)),                                                                    # 
        in_specs = (pl.BlockSpec(scalars.shape, lambda i, j, k: (0,), memory_space = pltpu.SMEM),                                     # for scalars 
                    pl.BlockSpec((bbt, bv), lambda i, j, k: (k, i), memory_space = pltpu.VMEM),                                       # for go block
                    pl.BlockSpec((bbt, bh), lambda i, j, k: (k, j), memory_space = pltpu.VMEM),                                       # for x block
                    pl.BlockSpec((bv, bh), lambda i, j, k: (i, j), memory_space = pltpu.VMEM),                                        # for w block
                    {n: codec.specs(bv, bh) for n in opt.states}),                                                                    # for states
        out_specs = (pl.BlockSpec((bv, bh), lambda i, j, k: (i, j), memory_space = pltpu.VMEM),                                       # for w_new
                     {n: codec.specs(bv, bh) for n in opt.states}),                                                                   # for states_new
        out_shape = _utils.like(w, states),                                                                                           #
        scratch_shapes = [pltpu.VMEM((bv, bh), jnp.float32)],                                                                         # for dw, one scratch_pad per pallas_call
        input_output_aliases = {3 + i: i for i in range(1 + _utils.count_arrays(states))},                                            # to overwrite w and states in place
        compiler_params = _utils.compiler_params(("parallel", "parallel", "arbitrary")),                                              # i, j can be done parallely, k must be sequential 0, 1, 2 ...
        interpret = interpret,                                                                                                        #
    )(scalars, go, x, w, states)                                                                                                      #



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def _kernel(scalars_ref, go_ref, x_ref, w_ref, state_refs, w_out, states_out, acc_ref, *, opt, codec, BT):
    # this is the main kernel function which updates the small (bv, bh) w and state blocks. how data moves from VMEM to VREGS is handled by compiler, we dont have to worry
    # go_ref[...] is (bbt, bv)   :   x_ref[...] is (bbt, bh)   :   w_ref[...] is (bv, bh) #
    k = pl.program_id(2)                                                                                                              #
    @pl.when(k == 0)                                                                                                                  #
    def autoCall(): acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)                                                              # replace garbage with zeros 
    go, x = go_ref[...], x_ref[...]                                                                                                   # moves from VMEM to vregs (not all at once ofcrs)
    bbt = go.shape[0]                                                                                                                 # NOTE : go_ref is a mutable ref<VMEM>(dtype) go_ref[...] is a jax.Array
    if BT % bbt:                                                                                                                      # 
        rows = k * bbt + jax.lax.broadcasted_iota(jnp.int32, (bbt, 1), 0)                                                             # go_ref and x_ref grab exactly (bbt, bv) and (bbt, bh) sized blocks
        go = jnp.where(rows < BT, go, 0)                                                                                              # and if BT % bbt then some are garbage values, this code block sets
        x = jnp.where(rows < BT, x, 0)                                                                                                # those garbage to 0
    acc_ref[...] += jax.lax.dot_general(                                                                                              # 
        go, x, (((0,), (0,)), ((), ())), preferred_element_type = jnp.float32,                                                        # accumulate go.T@x to scratch_pad
        precision = jax.lax.Precision.HIGHEST if go.dtype == jnp.float32 else None)                                                   # v5e- supports only int8 and bf16 natively, this says if fp32 then do it
    @pl.when(k == pl.num_programs(2) - 1)                                                                                             # precisely splitting it into bf16, but dont lose precision 
    def autoCall(): _epilogue(opt, codec, scalars_ref, w_ref, state_refs, acc_ref[...], w_out, states_out)                            # done accumulating for (i, j, _) now update



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def _epilogue(opt, codec, scalars_ref, w_ref, state_refs, grad, w_out, states_out):
    st = {state_string: codec.decode(state_refs[state_string], opt.floors.get(state_string)) for state_string in opt.states}
    w, st = opt.update(w_ref[...].astype(jnp.float32), grad, st, scalars_ref)                                                         # inputs to opt.update are all jax arrays of small tiles
    w_out[...] = w.astype(w_out.dtype)                                                                                                # NOTE : none of these functions are creating duplicates inside VMEM anywhere
    for state_string in opt.states: codec.encode(st[state_string], states_out[state_string])                                          # everything that seems like a copy (astype, decode, update, encode) is on a tile, in VREGs



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#   optimizer_only is commented : nothing calls it and it was never run on the TPU. Dikshant uses his optimizer-only kernel for two things, both doable on TPU :
#      1. gradient accumulation : dw of the micro-steps is added in a buffer, then one update with the full dw (see autograd.py)
#      2. use_fused_backward = False for a layer where the fused kernel is slower (he says : a very large V, because x is read again for every weight tile)
# @functools.partial(jax.jit, static_argnames = ("optimizer", "state_mode", "interpret", "tiles"))
# def optimizer_only(grad, w, states, scalars, *, optimizer, state_mode, interpret, tiles = None):
#     # Like fused_grad, with grad (V, H) given (any float dtype, each tile is used as fp32); bbt in tiles is ignored                   # 
#     opt, codec = _utils.lookup(optimizer, state_mode)                                                                                 # 
#     V, H = w.shape                                                                                                                    # 
#     bv, bh = _utils.pick_tiles(V, H, tiles, codec)                                                                                    # 
#     def kernel(s_ref, g_ref, w_ref, st_refs, w_out, st_out): _epilogue(opt, codec, s_ref, w_ref, st_refs, g_ref[...].astype(jnp.float32), w_out, st_out)
#     return pl.pallas_call(                                                                                                            # 
#         kernel, grid = (pl.cdiv(V, bv), pl.cdiv(H, bh)),                                                                              # 
#         in_specs = (pl.BlockSpec(scalars.shape, lambda i, j: (0,), memory_space = pltpu.SMEM),                                        # 
#                     pl.BlockSpec((bv, bh), lambda i, j: (i, j), memory_space = pltpu.VMEM),                                           # 
#                     pl.BlockSpec((bv, bh), lambda i, j: (i, j), memory_space = pltpu.VMEM),                                           # 
#                     {n: codec.specs(bv, bh) for n in opt.states}),                                                                    # 
#         out_specs = (pl.BlockSpec((bv, bh), lambda i, j: (i, j), memory_space = pltpu.VMEM),                                          # 
#                      {n: codec.specs(bv, bh) for n in opt.states}),                                                                   # 
#         out_shape = _utils.like(w, states),                                                                                           # 
#         input_output_aliases =  {2 + i: i for i in range(1 + _utils.count_arrays(states))},                                           # 
#         compiler_params = _utils.compiler_params(("parallel", "parallel")),                                                           # 
#         interpret = interpret,                                                                                                        # 
#     )(scalars, grad, w, states)                                                                                                       # 



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
