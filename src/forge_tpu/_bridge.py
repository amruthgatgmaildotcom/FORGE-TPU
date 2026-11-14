import functools, jax, torch, warnings
import jax.numpy as jnp
from forge_tpu import kernel

_TORCH = {"bfloat16": torch.bfloat16, "float32": torch.float32, "int8": torch.int8}
_JAX = {torch_type: type_string for type_string, torch_type in _TORCH.items()}
KICK_EVERY = 1                                                                                                                        # eager TPU : start the deferred work every KICK_EVERY updates, 0 : never (see _kick)
_COMPILED_OPS, _LAUNCHES, _IN_PLACE, _NO_REPLAY, _KICKS = {}, {}, set(), set(), [0]

#   Readme.md
#     we need a mediator between torch --- jax/pallas. This file contains fused_update function which recieves tensors and it internally calls the jax kernel via jax_op
#     also we have allocate_state function which will be used during the first forward to allocate state variables. 
#    MORE README AT THE END OF THE FILE
















#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def allocate_state(weight, optimizer_type, state_mode):
    # new state tensors for weight tensor, on its device, as {str : {str : tensor, }, }. Stored 2-D (V, H) as the kernels want them : a bias (n,) is (1, n)
    H = weight.shape[-1] if weight.dim() else 1
    spec = kernel.state_arrays(optimizer_type, state_mode, weight.numel() // H, H, jnp.dtype(_JAX[weight.dtype]))
    if weight.device.type == "tpu": _COMPILED_OPS.setdefault((optimizer_type, state_mode), _run_jax_op(optimizer_type, state_mode, donate = False))
    return {state_string: {stringg: torch.full(shape, fill, dtype = _TORCH[jnp.dtype(dt).name], device = weight.device) for stringg, (shape, dt, fill) in arrays.items()} for state_string, arrays in spec.items()}

def fused_update(go2d, x2d, fusedStateObject, with_dx):
    # go2d (BT, V), x2d (BT, H) tensors. Updates w and its state in place, returns dx (BT, H) computed with the old w (or None)       #
    w = fusedStateObject.weight                                                                                                       #
    dx = go2d @ w.detach() if with_dx else None                                                                                       # FIRST, with the old w (the update donates it). Its own program, so freed as soon as autograd drops it
    flat = _flatten(fusedStateObject.buffers, fusedStateObject.optimizer_type)                                                        # 
    # if torch.compiler.is_compiling():                                                                                                 # inside torch.compile : the whole step becomes one XLA program
    #     # step is a device tensor here (a Python int would be frozen at trace time), so the scalars are computed on device, and the new
    #     # w and states are written back with copy_ (set_ cannot be traced). XLA sees kernel + copy_ in one program and writes in place
    #     scalars = kernel.OPTIMIZERS[fusedStateObject.optimizer_type].scalars(fusedStateObject.hparams, fusedStateObject.step_tensor)
    #     s = torch.stack([v.float() if torch.is_tensor(v) else torch.tensor(v, dtype = torch.float32, device = w.device) for v in scalars])
    #     out = _COMPILED_OPS[(fusedStateObject.optimizer_type, fusedStateObject.state_mode)](s, [go2d.detach(), x2d.detach()], w.detach(), flat)
    #     with torch.no_grad(): for old, new in zip([w, *flat], out): old.copy_(new)
    #     return dx
    scalars = kernel.OPTIMIZERS[fusedStateObject.optimizer_type].scalars(fusedStateObject.hparams, fusedStateObject.step)             # list of all scalars
    if w.device.type == "tpu":                                                                                                        # dlpack doesnt support TPU, it only supports NVIDIA and CPU 
        s, key = _device_scalars(tuple(scalars), w.device), (fusedStateObject.optimizer_type, fusedStateObject.state_mode, tuple(w.shape), w.dtype, tuple(go2d.shape), go2d.dtype, x2d.dtype)
        inputs = [s, go2d, x2d, w, *flat]                                                                                             # the kernel's tensors, in torch_tpu's launch order (no detach : the backward runs without grad mode)
        call = lambda: _run_jax_op(fusedStateObject.optimizer_type, fusedStateObject.state_mode)(s, [go2d.detach(), x2d.detach()], w.detach(), flat)
        launch = _LAUNCHES.get(key)                                                                                                   # 
        out = _replay(launch, inputs, key, call) if launch else _launch(call, inputs, [w, *flat], key)                                # 
        _rebind(fusedStateObject, out); _kick(out[0])                                                                                 # w's new value
        return dx                                                                                                                     #
    elif w.device.type == "cpu": out = [_to_torch(a) for a in _run(jnp.asarray(scalars, jnp.float32), [_to_jax(go2d), _to_jax(x2d)], _to_jax(w), [_to_jax(t) for t in flat], optimizer = fusedStateObject.optimizer_type, state_mode = fusedStateObject.state_mode, interpret = True)]
    else: raise RuntimeError(f"FusedLinear supports devices 'tpu' and 'cpu', got '{w.device}'")                                       # 
    _rebind(fusedStateObject, out)                                                                                                    # make fusedStateObject point to the right data
    return dx



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def _flatten(tree, optimizer):
    # {str : {str : array}} -> list (jax_op takes lists, not dicts) : states in the optimizer's order, arrays in stored order         #
    return [tree[state_string][stringg] for state_string in kernel.OPTIMIZERS[optimizer].states for stringg in tree[state_string]]    #

def _unflatten(flat, template, optimizer):
    it = iter(flat)                                                                                                                   # the reverse, with template's keys
    return {state_string: {stringg: next(it) for stringg in template[state_string]} for state_string in kernel.OPTIMIZERS[optimizer].states}

def _to_jax(torchTensor):
    return jax.dlpack.from_dlpack(torchTensor.detach())                                                                               # this if for CPU path

def _to_torch(jaxArray):
    return torch.from_dlpack(jaxArray)                                                                                                # this is for CPU path

def _rebind(fusedStateObject, out):
    with torch.no_grad(): fusedStateObject.weight.set_(out[0])                                                                        # this function is to make fusedStateObject point at the right data. 
    new = iter(out[1:])                                                                                                               # The state dict is updated in place,
    for state_string in kernel.OPTIMIZERS[fusedStateObject.optimizer_type].states:                                                    # in _flatten's order (no new dict per call)
        arrays = fusedStateObject.buffers[state_string]                                                                               #
        for stringg in arrays: arrays[stringg] = next(new)                                                                            #
    # when we donate_argnums in jax_op, it unfortunately marks those tensors as deleted (those TPU pointers cannot be used) and fusedObject.members point no where now although the updates happen in place.
    # that is why we need to set_ these tensor objects. the problem with this is the set_ is not tracable, so we cannot use torch.compile as of now



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def _run(scalars: jax.Array, inputs: list[jax.Array], w: jax.Array, states: list[jax.Array], *, optimizer: str, state_mode: str, interpret: bool) -> tuple[jax.Array, ...]:
    # pure JAX. inputs = [go (BT, V), x (BT, H)], w (V, H), states = _flatten(buffers). Returns (w_new, *states_new)                  #
    template = kernel.state_arrays(optimizer, state_mode, 64, 128, jnp.float32)                                                       # only its keys are used
    w_new, states_new = kernel.fused_grad(*inputs, w, _unflatten(states, template, optimizer), scalars, optimizer = optimizer, state_mode = state_mode, interpret = interpret)
    return (w_new, *_flatten(states_new, optimizer))                                                                                  #

@functools.cache
def _run_jax_op(optimizer, state_mode, donate = True):
    # wrap _run (a pure jax, takees in and out jax.Array) with torch.tpu.pallas.jax_op, now _run_jax_op is basically _run but for torch tensors instead of jax arrays
    # donate = False is for torch.compile (see fused_update) : there w and states are written back with copy_, so they must not be given away
    name, donated = f"pallas::forge_fused_grad_{optimizer}_{state_mode}" + ("" if donate else "_compiled"), {"donate_argnums": (2, 3)} if donate else {}
    return torch.tpu.pallas.jax_op(name, functools.partial(_run, optimizer = optimizer, state_mode = state_mode, interpret = False), **donated)

@functools.lru_cache(maxsize = 8)
def _device_scalars(values, device):
    return torch.tensor(values, dtype = torch.float32, device = device)                                                               #



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def _in_place_or_raise(call, donated, key):
    # runs call() (the TPU update). The FIRST time for each key only : if XLA copied a donated tensor (w or a state) instead of writing into it, the TPU's peak memory grows by that tensor's size. A growth of half of a large donated tensor (>= 4 MiB : w, a bf16 state, an int8
    # q array) -> raise. A smaller growth (>= 1 MiB : a small array such as an int8 scale, or runtime memory) -> one warning, the update is kept. The same compiled program runs every later time : no check, no sync.
    if key in _IN_PLACE: return call()                                                                                                #
    sizes = sorted(t.nbytes for t in donated)                                                                                         #
    large = [b for b in sizes if b >= 4 * 2**20] or sizes                                                                             # small layers : all their tensors
    limit = max(large[0] // 2, 2**20)                                                                                                 # half the smallest large donated tensor, at least 1 MiB (noise)
    torch.accelerator.synchronize(); before = torch.accelerator.memory_allocated(); torch.accelerator.reset_peak_memory_stats()       #
    out = call()                                                                                                                      #
    torch.accelerator.synchronize(); extra = torch.accelerator.max_memory_allocated() - before                                        #
    report = f"FORGE update {key}: the TPU's peak memory grew by {extra / 2**20:.1f} MiB during it (donated tensors, MiB: {', '.join(f'{b / 2**20:.1f}' for b in sizes)})"
    if extra >= limit: raise RuntimeError(report + ", so XLA copied a large donated buffer instead of writing into it (NOT in place on HBM)")
    if extra >= 2**20: warnings.warn(report + ": less than half of any large donated tensor; a small array may have been copied, the update is kept")
    _IN_PLACE.add(key)                                                                                                                #
    return out                                                                                                                        #

def _launch(call, inputs, donated, key):
    from torch_tpu._internal.pallas import pallas                                                                                     #
    module, caught = pallas.tpu_torch_pallas, {}                                                                                      #
    class _Catch:                                                                                                                     # stands in for torch_tpu's C++ module during call()
        def __getattr__(self, name): return getattr(module, name)                                                                     #
        def call_custom_kernel(self, *args, **kwargs): caught.update(args = args, kwargs = kwargs); return module.call_custom_kernel(*args, **kwargs)
    pallas.tpu_torch_pallas = _Catch()                                                                                                #
    try: out = _in_place_or_raise(call, donated, key)                                                                                 #
    finally: pallas.tpu_torch_pallas = module                                                                                         #
    seen = caught.get("kwargs", {}).get("inputs", [])                                                                                 #
    if key not in _NO_REPLAY and len(seen) == len(inputs) and all(a.shape == b.shape and a.dtype == b.dtype for a, b in zip(seen, inputs)): _LAUNCHES[key] = (module.call_custom_kernel, caught["args"], {name: value for name, value in caught["kwargs"].items() if name != "inputs"}) 
    return out                                                                                                                        #

def _replay(launch, inputs, key, call):
    function, args, kwargs = launch                                                                                                   #
    try: return function(*args, inputs = inputs, **kwargs)                                                                            #
    except (TypeError, AttributeError) as error:                                                                                      #
        del _LAUNCHES[key]; _NO_REPLAY.add(key)                                                                                       #
        warnings.warn(f"FORGE: replaying torch_tpu's kernel launch failed ({error}); {key} goes through jax_op from now on (slower)") #
        return call()                                                                                                                 #

def _kick(weight):
    _KICKS[0] += 1                                                                                                                    # 
    if _KICKS[0] % KICK_EVERY == 0:                                                                                                   # 
        from torch_tpu._internal import sync                                                                                          # 
        sync.synchronize([weight], False)                                                                                             # wait = False : materialize, do not block



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#








#   WHAT HAPPENS IN torch.tpu.pallas.jax_op AND HOW TO OPTIMIZE IT WITH _launch :
#     _bridge.fused_update(...) ---> _run_jax_op(...) ---> torch.library.custom_op(...) ---> _JaxCustomOpDef.__call__ ---> tpu_torch_pallas.call_custom_kernel
#     The 5th one is our real kernel and takes very less time (about 21us for one llama-1B layer) while 3rd and 4th are just pytorch checks and is skippable.
#     (3rd + 4th layer took about 160us per llama-1B layer). _launch does the exact same thing torch would do and _replay skips 3rd and 4th and directly does 5th. 
#
#   TORCH_TPU_DEFER_AND_FUSE, STRICT EAGER, COMPILATION EXECUTUION FOR TPUs :
#     each tpu operation is compiled and cached, and we call an operation and wait until it returns, this is strict eager mode (default). This adds TPU launch 
#     overhead and its a lot. TORCH_TPU_DEFER_AND_FUSE = 1 would pile up the operations in a graph and execute them in groups (until one of them is needed to
#     return for someother part of the code). This mode has reduced no of TPU launches.
#     