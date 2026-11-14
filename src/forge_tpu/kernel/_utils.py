import jax
from jax.experimental.pallas import tpu as pltpu
from .codecs import CODECS
from .optimizers import OPTIMIZERS

DEFAULT_TILES = (512, 1024, 1024)     # (bv, bh, bbt)
# Dikshant autotunes the tiles for every (BT, V, H) (triton.autotune). NOT DONE HERE : one fixed triple for every layer. benchmarks/tiler.py times many triples for each
# layer shape of the benchmark models on the TPU; a table {(BT, V, H) : tiles} read by pick_tiles would do what his autotune does
#   Readme.md
#      This file is internal to the kernel folder (starts with _), used by fused_grad.py (and lookup by __init__.py)
#      each defined function is readable and self explainatory, so im skipping explaination here




















#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def lookup(optimizer, state_mode):
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {optimizer!r}; choose from {sorted(OPTIMIZERS)}")
    if state_mode not in CODECS:
        raise ValueError(f"unsupported state_mode {state_mode!r}; choose from {sorted(CODECS)}")
    return OPTIMIZERS[optimizer], CODECS[state_mode]

def _check(name, block, dim, multiple):
    if block != dim and (block <= 0 or block % multiple):
        raise ValueError(f"tile {name}={block} must equal {dim} or be a positive multiple of {multiple}")
    # DOCUMENTATION 
    # TPUs perform the bulk of the computation on 2D vector registers, which are typically of size 8x128 for 32-bit values
    # (as of TPU v6). When a vector value is loaded from VMEM into registers (e.g. x = x_ref[...]), the last two dimensions
    # of the array will be tiled into the registers. Pallas will only ever consider mapping the last two dimensions of
    # intermediate arrays to the 8x128 vector register dimensions (sublanes and lanes respectively). 
    # This holds for v5 also ==> (bv % 128 = 0, bh % 128 = 0, bbt % 8 = 0)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def pick_tiles(V, H, tiles, codec):
    bv, bh = tiles[:2] if tiles else (min(V, DEFAULT_TILES[0]), min(H, DEFAULT_TILES[1]))
    _check("bv", bv, V, 128); _check("bh", bh, H, 128); codec.check(V, bv)
    return bv, bh

def pick_bt_tile(BT, tiles):
    bbt = tiles[2] if tiles else min(BT, DEFAULT_TILES[2])
    _check("bbt", bbt, BT, 8)
    return bbt



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def like(w, states):
    # pallas_call's out_shape: the shape and dtype (no data) of w and of every state array.
    def shape_of(a): return jax.ShapeDtypeStruct(a.shape, a.dtype)
    return shape_of(w), {s: {name: shape_of(a) for name, a in arrays.items()} for s, arrays in states.items()}

def count_arrays(states):
    # This function returns totally how many state matrices are we maintaining 
    return sum(len(arrays) for arrays in states.values())

def compiler_params(semantics):
    # v5e caps a kernel's VMEM at 16 MiB by default, (128 MiB exist), setting it to 64MiB (MAX).
    return pltpu.CompilerParams(dimension_semantics = semantics, vmem_limit_bytes = 64*(2**20))



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

