"""How much host time can FORGE's eager Python still lose?  Whole steps only, no timing wrappers (they add their own cost).

A stack of DEPTH Llama-sized layers (2048 -> 8192 -> 2048 -> ...), bf16, batch 1 x SEQ tokens, eager (run it with
TORCH_TPU_DEFER_AND_FUSE=1, as the benchmark). Variants, the same layers and data:
    nn.Linear, no step      forward + backward only: the floor of torch's own host cost (it makes every dW, updates nothing)
    fused AdamW             nn.Linear, then torch.optim.AdamW(fused=True).step()
    FORGE (src)             FusedLinear as in src
    FORGE lean update       _bridge.fused_update replaced (only in this file) by one with less Python: no detach() (the
                            backward runs without grad mode), the update key cached per layer and go shape, the replay inline
    FORGE lean update+bwd   and autograd's _linear_backward + _apply_fused merged into one function (fewer Python calls)
    FORGE lean, no kick     lean update+bwd without _kick: what the kick costs on the host (and on the step)
The FORGE variants must equal FORGE (src)'s weights and states bit for bit ("max |diff|" = 0).
Per variant: step ms (median, after a sync), host ms (median, Python done), host us per FORGE layer above "nn.Linear, no
step", TPU programs per step, peak GiB.

    TORCH_TPU_DEFER_AND_FUSE=1 PYTHONPATH=src python benchmarks/probe_eager.py
"""
import os, statistics, time, torch, torch.nn as nn
from forge_tpu import _bridge, autograd, kernel

DEPTH, SEQ, HIDDEN, MLP, LR, WARMUP, TIMED = 16, 512, 2048, 8192, 1e-4, 5, 30
ON_TPU = torch.accelerator.is_available() and torch.accelerator.current_accelerator().type == "tpu"
DEV = "tpu" if ON_TPU else "cpu"
SRC_UPDATE, SRC_BACKWARD = _bridge.fused_update, autograd._linear_backward


def programs():
    return torch.tpu._get_cache_requests() if ON_TPU else 0

def sync():
    if ON_TPU: torch.accelerator.synchronize()


# ---- the lean variants (same arguments and results as src's functions) ----
KEYS = {}                                                                                                # (id(state), go shape) -> src's update key
KICK = [True]

def update_lean(go2d, x2d, st, with_dx):
    w = st.weight
    dx = go2d @ w if with_dx else None                                                                   # FIRST, with the old w
    launch = _bridge._LAUNCHES.get(KEYS.get((id(st), go2d.shape)))
    if launch is None:                                                                                   # first call: src's way (checks and keeps the launch)
        SRC_UPDATE(go2d, x2d, st, False)
        KEYS[id(st), go2d.shape] = (st.optimizer_type, st.state_mode, tuple(w.shape), w.dtype, tuple(go2d.shape), go2d.dtype, x2d.dtype)
        return dx
    from torch_tpu._internal.pallas import pallas
    s = _bridge._device_scalars(tuple(kernel.OPTIMIZERS[st.optimizer_type].scalars(st.hparams, st.step)), w.device)
    args, kwargs = launch
    out = pallas.tpu_torch_pallas.call_custom_kernel(*args, inputs = [s, go2d, x2d, w, *_bridge._flatten(st.buffers, st.optimizer_type)], **kwargs)
    _bridge._rebind(st, out)
    if KICK[0]: _bridge._kick(out[0])
    return dx

def backward_lean(grad_output, x, weight, has_bias, needs_dx, st):
    go2d = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    st.step += 1                                                                                         # eager: the Python int (increment_step)
    dx = _bridge.fused_update(go2d, x2d, st, needs_dx)
    return None if dx is None else dx.reshape(*grad_output.shape[:-1], -1), None, go2d.sum(0) if has_bias else None

VARIANTS = {"FORGE (src)": (SRC_UPDATE, SRC_BACKWARD, True), "FORGE lean update": (update_lean, SRC_BACKWARD, True),
            "FORGE lean update+bwd": (update_lean, backward_lean, True), "FORGE lean, no kick": (update_lean, backward_lean, False)}


# ---- the model and one variant ----
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

def run(name):
    fused = name.startswith("FORGE")
    if fused: _bridge.fused_update, autograd._linear_backward, KICK[0] = VARIANTS[name]
    model = stack(fused)
    optimizer = torch.optim.AdamW(model.parameters(), lr = LR, fused = True) if name == "fused AdamW" else None
    x = torch.randn(SEQ, HIDDEN, generator = torch.Generator().manual_seed(1)).to(DEV, torch.bfloat16).requires_grad_()
    rows = []
    for step in range(WARMUP + TIMED):
        sync(); p0 = programs()
        if ON_TPU: torch.accelerator.reset_peak_memory_stats()
        start = time.perf_counter()
        model(x).float().square().mean().backward()
        if optimizer: optimizer.step()
        for p in model.parameters(): p.grad = None
        x.grad = None
        host = time.perf_counter()
        sync()
        end = time.perf_counter()
        peak = torch.accelerator.max_memory_allocated() / 2**30 if ON_TPU else 0.0
        if step >= WARMUP: rows.append((1e3 * (end - start), 1e3 * (host - start), programs() - p0, peak))
    _bridge.fused_update, autograd._linear_backward = SRC_UPDATE, SRC_BACKWARD
    final = [t.detach().float().cpu() for l in model for t in [l.weight] + ([v for a in l.fusedStateObject.buffers.values() for v in a.values()] if fused else [])]
    return rows, final


if __name__ == "__main__":
    print(f"device {DEV}, torch {torch.__version__}, TORCH_TPU_DEFER_AND_FUSE={os.environ.get('TORCH_TPU_DEFER_AND_FUSE', 'not set')}, "
          f"KICK_EVERY={_bridge.KICK_EVERY}, {DEPTH} layers ({HIDDEN} <-> {MLP}), {SEQ} tokens, bf16, {WARMUP} warmup + {TIMED} timed steps")
    names = ["nn.Linear, no step", "fused AdamW", *VARIANTS] if ON_TPU else ["nn.Linear, no step", "fused AdamW", "FORGE (src)"]
    print(f"\n{'variant':<23} {'step ms':>8} {'host ms':>8} {'host us/layer over floor':>25} {'programs':>9} {'peak GiB':>9} {'max |diff| vs src':>18}")
    floor, reference = None, None
    for name in names:
        rows, final = run(name)
        med = lambda i: statistics.median(r[i] for r in rows)
        if floor is None: floor = med(1)
        if name == "FORGE (src)": reference = final
        diff = f"{max((a - b).abs().max().item() for a, b in zip(final, reference)):18.3e}" if name.startswith("FORGE") and reference else f"{'':>18}"
        print(f"{name:<23} {med(0):8.2f} {med(1):8.2f} {1e3 * (med(1) - floor) / DEPTH:25.1f} {med(2):9.1f} {max(r[3] for r in rows):9.3f} {diff}", flush = True)
