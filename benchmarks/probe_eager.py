"""Where do FORGE's eager programs and host time go?  Two tables.

A stack of DEPTH Llama-sized layers (2048 -> 8192 -> 2048 -> ...), bf16, batch 1 x SEQ tokens, eager. Methods, the same layers:
    fused AdamW          nn.Linear, then torch.optim.AdamW(fused=True).step()
    FORGE (src)          FusedLinear as in src (after the first call per shape, torch_tpu's C++ launch is replayed)
    FORGE via jax_op     src, but every update goes through the jax_op custom op (src before the replay). Its weights and
                         states must equal FORGE (src)'s bit for bit ("max |diff|" = 0)
Table 1, per step: total ms (after a sync), host ms (Python done), TPU programs (torch.tpu._get_cache_requests), peak GiB.
Table 2, FORGE (src), per layer: the host time and programs of each piece, measured by wrapping the functions in this
process (a piece contains the pieces indented below it; "rest" = the parent minus its listed pieces).

    TORCH_TPU_DEFER_AND_FUSE=1 PYTHONPATH=src python benchmarks/probe_eager.py
"""
import collections, functools, os, statistics, time, torch, torch.nn as nn
from forge_tpu import _bridge, autograd, kernel, module, state

DEPTH, SEQ, HIDDEN, MLP, LR, WARMUP, TIMED = 4, 512, 2048, 8192, 1e-4, 5, 10
ON_TPU = torch.accelerator.is_available() and torch.accelerator.current_accelerator().type == "tpu"
DEV = "tpu" if ON_TPU else "cpu"
METHODS = ["fused AdamW", "FORGE (src)", "FORGE via jax_op"] if ON_TPU else ["fused AdamW", "FORGE (src)"]


def programs():
    return torch.tpu._get_cache_requests() if ON_TPU else 0

def sync():
    if ON_TPU: torch.accelerator.synchronize()


# ---- timing wrappers (host ns and programs of every call, by method and name) ----
SPENT = collections.defaultdict(list)                                                                   # (method, name) -> [(host ns, programs)]
RECORD = [None]                                                                                          # the method being recorded, or None

def timed(name, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if RECORD[0] is None: return fn(*args, **kwargs)
        p, t = programs(), time.perf_counter_ns()
        out = fn(*args, **kwargs)
        SPENT[RECORD[0], name].append((time.perf_counter_ns() - t, programs() - p))
        return out
    return wrapper

def instrument():
    module.FusedLinear.forward = timed("forward", module.FusedLinear.forward)
    state.FusedOptimizerState.ensure_buffers = timed("ensure_buffers", state.FusedOptimizerState.ensure_buffers)
    apply = autograd.FusedLinearFunction.apply
    autograd.FusedLinearFunction.apply = timed("FusedLinearFunction.apply", apply)                       # FusedLinear.forward reads it from the class
    autograd.FusedLinearFunction.forward = staticmethod(timed("Function.forward", autograd.FusedLinearFunction.forward))
    autograd.FusedLinearFunction.backward = staticmethod(timed("backward", autograd.FusedLinearFunction.backward))
    _bridge.fused_update = timed("fused_update", _bridge.fused_update)
    adamw = kernel.OPTIMIZERS["adamw"]
    adamw.scalars = staticmethod(timed("scalars (list)", adamw.scalars))
    _bridge._device_scalars = timed("_device_scalars", _bridge._device_scalars)
    _bridge._flatten = timed("_flatten", _bridge._flatten)
    _bridge._launch = timed("_launch", _bridge._launch)
    _bridge._rebind = timed("_rebind", _bridge._rebind)

JAX_OP_ONLY = [False]
_LAUNCH = _bridge._launch                                                                                # src's, before instrument()
def _launch_or_jax_op(call, inputs, donated, key):
    return call() if JAX_OP_ONLY[0] else _LAUNCH(call, inputs, donated, key)
_bridge._launch = _launch_or_jax_op


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
    JAX_OP_ONLY[0] = method == "FORGE via jax_op"
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


PIECES = [("forward", 0, ["ensure_buffers", "FusedLinearFunction.apply"]), ("ensure_buffers", 2, []), ("FusedLinearFunction.apply", 2, ["Function.forward"]),
          ("Function.forward", 4, []), ("backward", 0, ["fused_update"]), ("fused_update", 2, ["scalars (list)", "_device_scalars", "_flatten", "_launch", "_rebind"]),
          ("scalars (list)", 4, []), ("_device_scalars", 4, []), ("_flatten", 4, []), ("_launch", 4, []), ("_rebind", 4, [])]

if __name__ == "__main__":
    print(f"device {DEV}, torch {torch.__version__}, TORCH_TPU_DEFER_AND_FUSE={os.environ.get('TORCH_TPU_DEFER_AND_FUSE', 'not set')}, "
          f"{DEPTH} layers ({HIDDEN} <-> {MLP}), {SEQ} tokens, bf16, {WARMUP} warmup + {TIMED} timed steps")
    instrument()
    print(f"\n{'method':<17} {'total ms':>9} {'host ms':>8} {'programs/step':>14} {'peak GiB':>9} {'max |diff| vs FORGE (src)':>26}")
    reference = None
    for method in METHODS:
        rows, final = run(method)
        if method == "FORGE (src)": reference = final
        diff = max((a - b).abs().max().item() for a, b in zip(final, reference)) if method.startswith("FORGE") else float("nan")
        med = lambda i: statistics.median(r[i] for r in rows)
        print(f"{method:<17} {med(0):9.2f} {med(1):8.2f} {med(2):14.1f} {max(r[3] for r in rows):9.3f} {diff:26.3e}", flush = True)
    calls = TIMED * DEPTH
    mean = lambda name: (sum(t for t, _ in SPENT[("FORGE (src)", name)]) / calls / 1e3, sum(p for _, p in SPENT[("FORGE (src)", name)]) / calls)
    print(f"\nFORGE (src), per layer (mean over {calls} layer calls; a piece contains the pieces indented below it):")
    print(f"{'piece':<34} {'calls/layer':>11} {'host us':>9} {'programs':>9}")
    for name, indent, children in PIECES:
        n = len(SPENT[("FORGE (src)", name)]) / calls
        us, prog = mean(name)
        print(f"{' ' * indent}{name:<{34 - indent}} {n:11.2f} {us:9.1f} {prog:9.2f}")
        if children:
            print(f"{' ' * (indent + 2)}{'rest':<{32 - indent}} {'':>11} {us - sum(mean(c)[0] for c in children):9.1f} {prog - sum(mean(c)[1] for c in children):9.2f}")
