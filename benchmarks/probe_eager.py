"""Where do FORGE's eager programs and host time go?  Two tables.

A stack of DEPTH Llama-sized layers (2048 -> 8192 -> 2048 -> ...), bf16, batch 1 x SEQ tokens, eager. Methods, the same layers:
    fused AdamW          nn.Linear, then torch.optim.AdamW(fused=True).step()
    FORGE (src)          FusedLinear as in src (after the first call per shape, torch_tpu's C++ launch is replayed)
    FORGE via jax_op     src, but every update goes through the jax_op custom op (src before the replay)
    FORGE lean-3         src's update with less Python around the launch (only in this file): no detach() (backward runs
                         without grad mode), the update key cached per layer and go shape, the replay called inline
The FORGE rows must equal FORGE (src)'s weights and states bit for bit ("max |diff|" = 0).
Table 1, per step: total ms (after a sync), host ms (Python done), TPU programs (torch.tpu._get_cache_requests), peak GiB.
Table 2, FORGE (src), per layer: the host time and programs of each piece, measured by wrapping the functions in this
process (a piece contains the pieces indented below it; "rest" = the parent minus its listed pieces). The wrappers add
a few us each to their parent's "rest".
Table 3, the host cost of single operations on a Llama-sized layer's tensors (mean over many calls, no wrappers).

    TORCH_TPU_DEFER_AND_FUSE=1 PYTHONPATH=src python benchmarks/probe_eager.py
"""
import collections, functools, os, statistics, time, torch, torch.nn as nn
from forge_tpu import _bridge, autograd, kernel, module, state

DEPTH, SEQ, HIDDEN, MLP, LR, WARMUP, TIMED = 4, 512, 2048, 8192, 1e-4, 5, 10
ON_TPU = torch.accelerator.is_available() and torch.accelerator.current_accelerator().type == "tpu"
DEV = "tpu" if ON_TPU else "cpu"
METHODS = ["fused AdamW", "FORGE (src)", "FORGE via jax_op", "FORGE lean-3"] if ON_TPU else ["fused AdamW", "FORGE (src)"]


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

SRC_FUSED_UPDATE, KEYS = _bridge.fused_update, {}                                                        # KEYS: (id(state), go shape) -> src's update key
def update_lean3(go2d, x2d, st, with_dx):
    w = st.weight
    dx = go2d @ w if with_dx else None                                                                   # FIRST, with the old w
    key = KEYS.get((id(st), go2d.shape))
    launch = _bridge._LAUNCHES.get(key)
    if launch is None:                                                                                   # first call: src's way (it checks and keeps the launch)
        out = SRC_FUSED_UPDATE(go2d, x2d, st, False)
        KEYS[id(st), go2d.shape] = (st.optimizer_type, st.state_mode, tuple(w.shape), w.dtype, tuple(go2d.shape), go2d.dtype, x2d.dtype)
        return dx
    s = _bridge._device_scalars(tuple(kernel.OPTIMIZERS[st.optimizer_type].scalars(st.hparams, st.step)), w.device)
    args, kwargs = launch
    from torch_tpu._internal.pallas import pallas
    out = pallas.tpu_torch_pallas.call_custom_kernel(*args, inputs = [s, go2d, x2d, w, *_bridge._flatten(st.buffers, st.optimizer_type)], **kwargs)
    _bridge._rebind(st, out)
    return dx


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
    if fused: _bridge.fused_update = timed("fused_update", update_lean3 if method == "FORGE lean-3" else SRC_FUSED_UPDATE)
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


# ---- Table 3: single operations ----
def cost(fn, n = 2000):
    for _ in range(min(n, 50)): fn()
    sync()
    t = time.perf_counter_ns()
    for _ in range(n): fn()
    return (time.perf_counter_ns() - t) / n / 1e3

def single_ops():
    w = torch.zeros(MLP, HIDDEN, dtype = torch.bfloat16, device = DEV, requires_grad = True)
    go, x = torch.zeros(SEQ, MLP, dtype = torch.bfloat16, device = DEV), torch.zeros(SEQ, HIDDEN, dtype = torch.bfloat16, device = DEV)
    go3 = go.reshape(1, SEQ, MLP)
    hp, step = dict(kernel.OPTIMIZERS["adamw"].defaults), [0]
    values = tuple(kernel.OPTIMIZERS["adamw"].scalars(hp, 1))
    def miss(): step[0] += 1; _bridge._device_scalars(tuple(kernel.OPTIMIZERS["adamw"].scalars(hp, step[0])), w.device)
    class Empty(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a): return a
        @staticmethod
        def backward(ctx, g): return g
    with torch.no_grad():
        rows = [("w.detach()", lambda: w.detach()), ("go3.reshape(-1, V)", lambda: go3.reshape(-1, MLP)), ("go.contiguous() (already)", lambda: go.contiguous()),
                ("w.device", lambda: w.device), ("w.device.type == 'tpu'", lambda: w.device.type == "tpu"), ("tuple(w.shape)", lambda: tuple(w.shape)),
                ("src's update key tuple", lambda: ("adamw", "bf16", tuple(w.shape), w.dtype, tuple(go.shape), go.dtype, x.dtype)),
                ("torch.compiler.is_compiling()", torch.compiler.is_compiling),
                ("AdamW.scalars(hp, step) list", lambda: kernel.OPTIMIZERS["adamw"].scalars(hp, 1)),
                ("_device_scalars, cache hit", lambda: _bridge._device_scalars(values, w.device)),
                ("_device_scalars, cache miss (new step)", miss),
                ("go @ w (issue only)", lambda: go @ w), ("F.linear(x, w) (issue only)", lambda: torch.nn.functional.linear(x, w)),
                ("w.forge_fused = True", lambda: setattr(w, "forge_fused", True))]
        out = []
        for name, fn in rows:                                                                            # matmuls: few calls and a sync after, so the deferred work stays small
            out.append((name, cost(fn, 20 if "issue" in name else 200 if "miss" in name else 2000))); sync()
    xg = torch.zeros(4, 4, device = DEV, requires_grad = True)
    out.append(("empty autograd.Function.apply (grad on)", cost(lambda: Empty.apply(xg))))
    sync()
    return out


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
    if ON_TPU:
        print(f"\nSingle operations (host us per call, mean):")
        for name, us in single_ops(): print(f"  {name:<42} {us:8.1f}")
