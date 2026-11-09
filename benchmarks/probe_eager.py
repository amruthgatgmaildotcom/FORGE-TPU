"""Where do FORGE's eager programs and host time go, and how much do two leaner kernel calls save?  Two tables.

A stack of DEPTH Llama-sized layers (2048 -> 8192 -> 2048 -> ...), bf16, batch 1 x SEQ tokens, eager. Methods, the same layers:
    fused AdamW    nn.Linear, then torch.optim.AdamW(fused=True).step()
    FORGE (bf16)   FusedLinear as in src
    FORGE lean-1   the same update, but the kernel is called through torch_tpu's JaxCallable directly (no torch custom-op
                   dispatch) and without the per-call key tuple / in-place check
    FORGE lean-2   lean-1, and after the first call per shape the C++ launch (call_custom_kernel) is replayed directly
                   with the arguments torch_tpu gave it (no JaxCallable Python), and the state dict is updated in place
The lean variants exist only in this file (src is not changed). Their weights and states must equal FORGE (bf16)'s
bit for bit ("max |diff|" = 0).
Table 1, per step: total ms (after a sync), host ms (Python done), TPU programs (torch.tpu._get_cache_requests), peak GiB.
Table 2, per layer backward: the host time and programs of each piece (a piece contains the pieces indented below it).

    TORCH_TPU_DEFER_AND_FUSE=1 PYTHONPATH=src python benchmarks/probe_eager.py
"""
import collections, functools, os, statistics, time, torch, torch.nn as nn
from forge_tpu import _bridge, autograd, kernel

DEPTH, SEQ, HIDDEN, MLP, LR, WARMUP, TIMED = 4, 512, 2048, 8192, 1e-4, 5, 10
ON_TPU = torch.accelerator.is_available() and torch.accelerator.current_accelerator().type == "tpu"
DEV = "tpu" if ON_TPU else "cpu"
METHODS = ["fused AdamW", "FORGE (bf16)", "FORGE lean-1", "FORGE lean-2"] if ON_TPU else ["fused AdamW", "FORGE (bf16)"]


def programs():
    return torch.tpu._get_cache_requests() if ON_TPU else 0

def sync():
    if ON_TPU: torch.accelerator.synchronize()


# ---- timing wrappers (host ns and programs of every call, by name) ----
SPENT = collections.defaultdict(list)                                                                   # (method, name) -> [(host ns, programs)]
RECORD = [None]                                                                                          # the method being recorded, or None
LAST_LAUNCH = {}                                                                                         # the last call_custom_kernel(*args, **kwargs)

def timed(name, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not RECORD[0]: return fn(*args, **kwargs)
        p, t = programs(), time.perf_counter_ns()
        out = fn(*args, **kwargs)
        SPENT[RECORD[0], name].append((time.perf_counter_ns() - t, programs() - p))
        return out
    return wrapper

class _Proxy:
    # stands in for the C++ module tpu_torch_pallas: call_custom_kernel is timed and its arguments kept, the rest is forwarded
    def __init__(self, module):
        self._module, launch = module, timed("call_custom_kernel", module.call_custom_kernel)
        def call_custom_kernel(*args, **kwargs):
            LAST_LAUNCH.update(args = args, kwargs = kwargs)
            return launch(*args, **kwargs)
        self.call_custom_kernel = call_custom_kernel
    def __getattr__(self, name): return getattr(self._module, name)

RUN_JAX_OP = _bridge._run_jax_op                                                                         # src's, before instrument()
SRC_FUSED_UPDATE = _bridge.fused_update

def instrument():
    autograd.FusedLinearFunction.backward = staticmethod(timed("backward", autograd.FusedLinearFunction.backward))
    _bridge._rebind = timed("_rebind", _bridge._rebind)
    _bridge._run_jax_op = functools.cache(lambda *a, **k: timed("custom op", RUN_JAX_OP(*a, **k)))
    if ON_TPU:
        from torch_tpu._internal.pallas import pallas
        pallas.JaxCallable.__call__ = timed("JaxCallable", pallas.JaxCallable.__call__)
        pallas.tpu_torch_pallas = _Proxy(pallas.tpu_torch_pallas)


# ---- the two lean updates (same arguments and result as _bridge.fused_update) ----
def _scalars(st, device):
    return _bridge._device_scalars(tuple(kernel.OPTIMIZERS[st.optimizer_type].scalars(st.hparams, st.step)), device)

def update_lean1(go2d, x2d, st, with_dx):
    w = st.weight
    dx = go2d @ w.detach() if with_dx else None
    jax_callable = RUN_JAX_OP(st.optimizer_type, st.state_mode)._init_fn                                 # the function inside the custom op
    out = jax_callable(_scalars(st, w.device), [go2d.detach(), x2d.detach()], w.detach(), _bridge._flatten(st.buffers, st.optimizer_type))
    _bridge._rebind(st, out)
    return dx

LAUNCHES = {}                                                                                            # (optimizer, state_mode, w shape, go shape) -> (args, kwargs) of call_custom_kernel

def update_lean2(go2d, x2d, st, with_dx):
    t0 = time.perf_counter_ns()
    w = st.weight
    dx = go2d @ w.detach() if with_dx else None
    t1 = time.perf_counter_ns()
    s, flat = _scalars(st, w.device), [t for arrays in st.buffers.values() for t in arrays.values()]
    key = (st.optimizer_type, st.state_mode, w.shape, go2d.shape)
    launch = LAUNCHES.get(key)
    if launch is None:                                                                                   # first call for this shape: through torch_tpu, keep its launch
        out = RUN_JAX_OP(st.optimizer_type, st.state_mode)._init_fn(s, [go2d.detach(), x2d.detach()], w.detach(), flat)
        LAUNCHES[key] = (LAST_LAUNCH["args"], {k: v for k, v in LAST_LAUNCH["kwargs"].items() if k != "inputs"})
    else:
        from torch_tpu._internal.pallas import pallas
        args, kwargs = launch
        out = pallas.tpu_torch_pallas._module.call_custom_kernel(*args, inputs = [s, go2d.detach(), x2d.detach(), w.detach(), *flat], **kwargs)
    t2 = time.perf_counter_ns()
    with torch.no_grad(): w.set_(out[0])
    rest = iter(out[1:])
    for arrays in st.buffers.values():
        for name in arrays: arrays[name] = next(rest)
    t3 = time.perf_counter_ns()
    if RECORD[0] and launch is not None:
        for name, ns in (("lean-2: dx matmul", t1 - t0), ("lean-2: scalars + launch", t2 - t1), ("lean-2: rebind", t3 - t2)):
            SPENT[RECORD[0], name].append((ns, 0))
    return dx

UPDATES = {"FORGE (bf16)": SRC_FUSED_UPDATE, "FORGE lean-1": update_lean1, "FORGE lean-2": update_lean2}


# ---- the model and one method ----
def stack(fused):
    from forge_tpu import FusedLinear
    torch.manual_seed(0)
    sizes = [(HIDDEN, MLP) if i % 2 == 0 else (MLP, HIDDEN) for i in range(DEPTH)]
    layers = [nn.Linear(i, o, bias = False, dtype = torch.bfloat16) for i, o in sizes]
    if fused: layers = [FusedLinear.from_linear(l, lr = LR) for l in layers]
    model = nn.Sequential(*layers).to(DEV).train()
    if fused:
        for l in model: l.fusedStateObject.ensure_buffers(l.weight)
    return model

def run(method):
    fused = method.startswith("FORGE")
    if fused: _bridge.fused_update = timed("fused_update", UPDATES[method])
    model = stack(fused)
    optimizer = None if fused else torch.optim.AdamW(model.parameters(), lr = LR, fused = True)
    x = torch.randn(SEQ, HIDDEN, generator = torch.Generator().manual_seed(1)).to(DEV, torch.bfloat16).requires_grad_()
    rows = []
    for step in range(WARMUP + TIMED):
        RECORD[0] = method if fused and step >= WARMUP else None
        sync(); p0 = programs()
        if ON_TPU: torch.accelerator.reset_peak_memory_stats()
        start = time.perf_counter()
        model(x).float().square().mean().backward()
        if optimizer: optimizer.step(); optimizer.zero_grad(set_to_none = True)
        x.grad = None
        host = time.perf_counter()
        sync()
        end = time.perf_counter()
        peak = torch.accelerator.max_memory_allocated() / 2**30 if ON_TPU else 0.0
        if step >= WARMUP: rows.append((1e3 * (end - start), 1e3 * (host - start), programs() - p0, peak))
    RECORD[0] = None
    final = [t.detach().float().cpu() for l in model for t in [l.weight] + ([v for a in l.fusedStateObject.buffers.values() for v in a.values()] if fused else [])]
    return rows, final


if __name__ == "__main__":
    print(f"device {DEV}, torch {torch.__version__}, TORCH_TPU_DEFER_AND_FUSE={os.environ.get('TORCH_TPU_DEFER_AND_FUSE', 'not set')}, "
          f"{DEPTH} layers ({HIDDEN} <-> {MLP}), {SEQ} tokens, bf16, {WARMUP} warmup + {TIMED} timed steps")
    instrument()
    print(f"\n{'method':<14} {'total ms':>9} {'host ms':>8} {'programs/step':>14} {'peak GiB':>9} {'max |diff| vs FORGE (bf16)':>27}")
    reference = None
    for method in METHODS:
        rows, final = run(method)
        if method == "FORGE (bf16)": reference = final
        diff = max((a - b).abs().max().item() for a, b in zip(final, reference)) if method.startswith("FORGE") else float("nan")
        med = lambda i: statistics.median(r[i] for r in rows)
        print(f"{method:<14} {med(0):9.2f} {med(1):8.2f} {med(2):14.1f} {max(r[3] for r in rows):9.3f} {diff:27.3e}", flush = True)
    calls = TIMED * DEPTH                                                                               # layer backwards recorded per method
    print(f"\nPer layer backward (mean over {calls} calls; a piece contains the pieces indented below it):")
    print(f"{'piece':<34} {'calls/layer':>11} {'host us':>9} {'programs':>9}")
    pieces = {"FORGE (bf16)": [("backward", 2), ("fused_update", 4), ("custom op", 6), ("JaxCallable", 8), ("call_custom_kernel", 10), ("_rebind", 6)],
              "FORGE lean-1": [("backward", 2), ("fused_update", 4), ("JaxCallable", 6), ("call_custom_kernel", 8), ("_rebind", 6)],
              "FORGE lean-2": [("backward", 2), ("fused_update", 4), ("lean-2: dx matmul", 6), ("lean-2: scalars + launch", 6), ("lean-2: rebind", 6)]}
    for method in METHODS[1:]:
        print(method)
        for name, indent in pieces[method]:
            got = SPENT.get((method, name), [])
            if got:
                print(f"{' ' * indent}{name:<{34 - indent}} {len(got) / calls:11.2f} {sum(t for t, _ in got) / len(got) / 1e3:9.1f} {sum(p for _, p in got) / len(got):9.2f}")
            else:
                print(f"{' ' * indent}{name:<{34 - indent}} {'not called':>11}")
