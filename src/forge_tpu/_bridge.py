import functools, jax, torch, warnings
import jax.numpy as jnp
from forge_tpu import kernel

_TORCH = {"bfloat16": torch.bfloat16, "float32": torch.float32, "int8": torch.int8}
_JAX = {torch_type: type_string for type_string, torch_type in _TORCH.items()}
KICK_EVERY = 1                                                                                                                        # eager TPU : start the deferred work every KICK_EVERY updates (see _kick)























#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def allocate_state(weight, optimizer_type, state_mode):
    # new state tensors for weight tensor, on its device, as {str : {str : tensor, }, }. Stored 2-D (V, H) as the kernels want them : a bias (n,) is (1, n)
    H = weight.shape[-1] if weight.dim() else 1
    spec = kernel.state_arrays(optimizer_type, state_mode, weight.numel() // H, H, jnp.dtype(_JAX[weight.dtype]))
    if weight.device.type == "tpu": _COMPILED_OPS.setdefault((optimizer_type, state_mode), _run_jax_op(optimizer_type, state_mode, donate = False))
    return {state_string: {stringg: torch.full(shape, fill, dtype = _TORCH[jnp.dtype(dt).name], device = weight.device) for stringg, (shape, dt, fill) in arrays.items()} for state_string, arrays in spec.items()}

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
    with torch.no_grad(): fusedStateObject.weight.set_(out[0])                                                                        # this function is to make fusedStateObject
    new = iter(out[1:])                                                                                                               # point at the right data. The state dict is updated in place,
    for state_string in kernel.OPTIMIZERS[fusedStateObject.optimizer_type].states:                                                    # in _flatten's order (no new dict per call)
        arrays = fusedStateObject.buffers[state_string]
        for stringg in arrays: arrays[stringg] = next(new)



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

_COMPILED_OPS = {}                                                                                                                    # (optimizer, state_mode) -> _run_jax_op(..., donate = False), made in allocate_state
                                                                                                                                      # (eager), because torch.compile cannot trace the creation of a jax_op
@functools.lru_cache(maxsize = 8)
def _device_scalars(values, device):
    return torch.tensor(values, dtype = torch.float32, device = device)                                                               #

_IN_PLACE = set()                                                                                                                     # (optimizer, state_mode, w shape and dtype, go shape and dtype, x dtype) whose update was seen in place

def _in_place_or_raise(call, donated, key):
    # runs call() (the TPU update). The FIRST time for each key only : if XLA copied a donated tensor (w or a state) instead of writing into
    # it, the TPU's peak memory grows by that tensor's size -> raise. The same compiled program runs every later time : no check, no sync.
    if key in _IN_PLACE: return call()
    limit = max(min(t.nbytes for t in donated) // 2, 2**20)                                                                           # half the smallest donated tensor, at least 1 MiB (noise)
    torch.accelerator.synchronize(); before = torch.accelerator.memory_allocated(); torch.accelerator.reset_peak_memory_stats()
    out = call()
    torch.accelerator.synchronize(); extra = torch.accelerator.max_memory_allocated() - before
    if extra >= limit: raise RuntimeError(f"FORGE update {key} was NOT in place on HBM: the TPU's peak memory grew by {extra / 2**20:.1f} MiB during it (w + states are {sum(t.nbytes for t in donated) / 2**20:.1f} MiB), so XLA copied a donated buffer instead of writing into it")
    _IN_PLACE.add(key)
    return out

_LAUNCHES = {}                                                                                                                        # key (as _IN_PLACE's) -> (args, kwargs) of torch_tpu's C++ kernel launch, kept at the key's first call

def _launch(call, inputs, donated, key):
    # runs the TPU update. The FIRST time for each key : call() (the jax_op, checked by _in_place_or_raise), while torch_tpu's C++ launch
    # is caught and kept. Every later time : that launch is replayed with the new inputs [s, go, x, w, *states], which skips ~200 us of
    # torch_tpu Python per layer (custom op dispatch + JaxCallable). call_custom_kernel is torch_tpu._internal (not a public API) : if
    # its signature ever changes, the replay is dropped with a warning and call() is used again.
    from torch_tpu._internal.pallas import pallas
    if key in _LAUNCHES:
        args, kwargs = _LAUNCHES[key]
        try: return pallas.tpu_torch_pallas.call_custom_kernel(*args, inputs = inputs, **kwargs)
        except (TypeError, AttributeError) as error:
            del _LAUNCHES[key]; _NO_REPLAY.add(key)
            warnings.warn(f"FORGE: replaying torch_tpu's kernel launch failed ({error}); {key} goes through jax_op from now on (slower)")
            return call()
    module, caught = pallas.tpu_torch_pallas, {}
    class _Catch:                                                                                                                     # stands in for torch_tpu's C++ module during call()
        def __getattr__(self, name): return getattr(module, name)
        def call_custom_kernel(self, *args, **kwargs): caught.update(args = args, kwargs = kwargs); return module.call_custom_kernel(*args, **kwargs)
    pallas.tpu_torch_pallas = _Catch()
    try: out = _in_place_or_raise(call, donated, key)
    finally: pallas.tpu_torch_pallas = module
    seen = caught.get("kwargs", {}).get("inputs", [])
    if key not in _NO_REPLAY and len(seen) == len(inputs) and all(a.shape == b.shape and a.dtype == b.dtype for a, b in zip(seen, inputs)):
        _LAUNCHES[key] = (caught["args"], {name: value for name, value in caught["kwargs"].items() if name != "inputs"})          # kept only if torch_tpu launched exactly our inputs
    return out

_NO_REPLAY = set()                                                                                                                    # keys whose replay failed once

_KICKS = [0]                                                                                                                          # eager TPU updates so far

def _kick(w):
    # starts the deferred work up to this update (w's new value) on the TPU, without waiting for it. With TORCH_TPU_DEFER_AND_FUSE
    # FORGE's backward otherwise stays deferred until the step's sync, so the TPU idles while Python runs the backward (Llama-3.2-1B,
    # v5e, seq 512 : 135.9 -> 88.1 ms per step with KICK_EVERY = 1 ; 7 : 109.3, 28 : 97.4). torch_tpu._internal (not a public API)
    _KICKS[0] += 1
    if _KICKS[0] % KICK_EVERY == 0:
        from torch_tpu._internal import sync
        sync.synchronize([w], False)                                                                                                  # wait = False : materialize, do not block

def fused_update(go2d, x2d, fusedStateObject, with_dx):
    # go2d (BT, V), x2d (BT, H) tensors. Updates w and its state in place, returns dx (BT, H) computed with the old w (or None)       #
    w = fusedStateObject.weight                                                                                                       #
    dx = go2d @ w.detach() if with_dx else None                                                                                       # FIRST, with the old w (the update donates it). Its own program, so freed as soon as autograd drops it
    flat = _flatten(fusedStateObject.buffers, fusedStateObject.optimizer_type)                                                        #
    if torch.compiler.is_compiling():                                                                                                 # inside torch.compile : the whole step becomes one XLA program
        # step is a device tensor here (a Python int would be frozen at trace time), so the scalars are computed on device, and the new
        # w and states are written back with copy_ (set_ cannot be traced). XLA sees kernel + copy_ in one program and writes in place.
        scalars = kernel.OPTIMIZERS[fusedStateObject.optimizer_type].scalars(fusedStateObject.hparams, fusedStateObject.step_tensor)
        s = torch.stack([v.float() if torch.is_tensor(v) else torch.tensor(v, dtype = torch.float32, device = w.device) for v in scalars])
        out = _COMPILED_OPS[(fusedStateObject.optimizer_type, fusedStateObject.state_mode)](s, [go2d.detach(), x2d.detach()], w.detach(), flat)
        with torch.no_grad():
            for old, new in zip([w, *flat], out): old.copy_(new)
        return dx
    scalars = kernel.OPTIMIZERS[fusedStateObject.optimizer_type].scalars(fusedStateObject.hparams, fusedStateObject.step)             # list of all scalars
    if w.device.type == "tpu":
        s, key = _device_scalars(tuple(scalars), w.device), (fusedStateObject.optimizer_type, fusedStateObject.state_mode, tuple(w.shape), w.dtype, tuple(go2d.shape), go2d.dtype, x2d.dtype)
        inputs = [s, go2d.detach(), x2d.detach(), w.detach(), *flat]                                                                  # the kernel's tensors, in torch_tpu's launch order
        out = _launch(lambda: _run_jax_op(fusedStateObject.optimizer_type, fusedStateObject.state_mode)(s, inputs[1:3], inputs[3], flat), inputs, [w, *flat], key)
    elif w.device.type == "cpu": out = [_to_torch(a) for a in _run(jnp.asarray(scalars, jnp.float32), [_to_jax(go2d), _to_jax(x2d)], _to_jax(w), [_to_jax(t) for t in flat], optimizer = fusedStateObject.optimizer_type, state_mode = fusedStateObject.state_mode, interpret = True)]
    else: raise RuntimeError(f"FusedLinear supports devices 'tpu' and 'cpu', got '{w.device}'")                                       #
    _rebind(fusedStateObject, out)                                                                                                    # make fusedStateObject point to the right data
    if w.device.type == "tpu": _kick(fusedStateObject.weight)                                                                         #
    return dx



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
