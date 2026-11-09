"""Where do FORGE's eager programs and host time go?  One short run, two tables.

A stack of DEPTH Llama-sized layers (2048 -> 8192 -> 2048 -> ...), bf16, batch 1 x SEQ tokens, eager (run it with
TORCH_TPU_DEFER_AND_FUSE=1, as the benchmark does). Two methods, the same layers:
    fused AdamW   nn.Linear, then torch.optim.AdamW(fused=True).step()
    FORGE (bf16)  FusedLinear, updated inside backward
Table 1, per step: total ms (after a sync), host ms (Python done), TPU programs (torch.tpu._get_cache_requests), peak GiB.
Table 2, FORGE only, per layer backward: the host time and the programs of each piece of the update, measured by
wrapping the functions in this process (src is not changed):
    backward          FusedLinearFunction.backward, everything
    fused_update      _bridge.fused_update (dx matmul + the kernel + the rebind)
    custom op         the torch.library op that jax_op made (dispatcher + everything below)
    JaxCallable       torch_tpu's Python wrapper of the kernel (argument checks, cache key, ...)
    call_custom_kernel  the C++ launch of the Pallas kernel
    _rebind           set_ of w, new state dict

    TORCH_TPU_DEFER_AND_FUSE=1 PYTHONPATH=src python benchmarks/probe_eager.py
"""
import collections, functools, os, statistics, time, torch, torch.nn as nn

DEPTH, SEQ, HIDDEN, MLP, LR, WARMUP, TIMED = 4, 512, 2048, 8192, 1e-4, 5, 10
ON_TPU = torch.accelerator.is_available() and torch.accelerator.current_accelerator().type == "tpu"
DEV = "tpu" if ON_TPU else "cpu"


def programs():
    return torch.tpu._get_cache_requests() if ON_TPU else 0

def sync():
    if ON_TPU: torch.accelerator.synchronize()


# ---- Table 2: timing wrappers (host ns and programs of every call, by name) ----
SPENT = collections.defaultdict(list)                                                                   # name -> [(host ns, programs)]
RECORD = [False]

def timed(name, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not RECORD[0]: return fn(*args, **kwargs)
        p, t = programs(), time.perf_counter_ns()
        out = fn(*args, **kwargs)
        SPENT[name].append((time.perf_counter_ns() - t, programs() - p))
        return out
    return wrapper

class _Proxy:
    # stands in for the C++ module tpu_torch_pallas: call_custom_kernel is timed, the rest is forwarded
    def __init__(self, module): self._module, self.call_custom_kernel = module, timed("call_custom_kernel", module.call_custom_kernel)
    def __getattr__(self, name): return getattr(self._module, name)

def instrument():
    from forge_tpu import _bridge, autograd
    autograd.FusedLinearFunction.backward = staticmethod(timed("backward", autograd.FusedLinearFunction.backward))
    _bridge.fused_update = timed("fused_update", _bridge.fused_update)
    _bridge._rebind = timed("_rebind", _bridge._rebind)
    original = _bridge._run_jax_op
    _bridge._run_jax_op = functools.cache(lambda *a, **k: timed("custom op", original(*a, **k)))
    if ON_TPU:
        from torch_tpu._internal.pallas import pallas
        pallas.JaxCallable.__call__ = timed("JaxCallable", pallas.JaxCallable.__call__)
        pallas.tpu_torch_pallas = _Proxy(pallas.tpu_torch_pallas)


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
    model = stack(fused)
    optimizer = None if fused else torch.optim.AdamW(model.parameters(), lr = LR, fused = True)
    x = torch.randn(SEQ, HIDDEN, generator = torch.Generator().manual_seed(1)).to(DEV, torch.bfloat16).requires_grad_()
    rows = []
    for step in range(WARMUP + TIMED):
        RECORD[0] = fused and step >= WARMUP
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
    RECORD[0] = False
    return rows


if __name__ == "__main__":
    print(f"device {DEV}, torch {torch.__version__}, TORCH_TPU_DEFER_AND_FUSE={os.environ.get('TORCH_TPU_DEFER_AND_FUSE', 'not set')}, "
          f"{DEPTH} layers ({HIDDEN} <-> {MLP}), {SEQ} tokens, bf16, {WARMUP} warmup + {TIMED} timed steps")
    instrument()
    print(f"\n{'method':<14} {'total ms':>9} {'host ms':>8} {'programs/step':>14} {'peak GiB':>9}")
    for method in ("fused AdamW", "FORGE (bf16)"):
        rows = run(method)
        med = lambda i: statistics.median(r[i] for r in rows)
        print(f"{method:<14} {med(0):9.2f} {med(1):8.2f} {med(2):14.1f} {max(r[3] for r in rows):9.3f}", flush = True)
    calls = TIMED * DEPTH                                                                               # FORGE layer backwards recorded
    print(f"\nFORGE, per layer backward (mean over {calls} calls; a piece contains the pieces below it):")
    print(f"{'piece':<20} {'calls/layer':>11} {'host us':>9} {'programs':>9}")
    for name in ("backward", "fused_update", "custom op", "JaxCallable", "call_custom_kernel", "_rebind"):
        got = SPENT.get(name, [])
        if got:
            print(f"{name:<20} {len(got) / calls:11.2f} {sum(t for t, _ in got) / len(got) / 1e3:9.1f} {sum(p for _, p in got) / len(got):9.2f}")
        else:
            print(f"{name:<20} {'not called':>11}")
