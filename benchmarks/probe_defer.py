"""Why does FORGE do more TPU work with TORCH_TPU_DEFER_AND_FUSE=1?  One layer, the update alone.

One Llama MLP-sized layer (w 8192 x 2048 bf16, bf16 states), go (512, 8192), x (512, 2048), fixed. Measured:
  1. TPU ms per update: N FORGE updates (_bridge.fused_update, no dx) and then one sync, against N fused-AdamW updates of
     the same weight (dW = go.T @ x, then torch.optim.AdamW(fused=True).step()). Also the peak memory above what was
     held before (an in-place update adds ~0; a copy of w + states adds ~96 MiB).
  2. With defer and fuse only: the deferred program (StableHLO) that one FORGE update makes, from
     torch_tpu._internal.sync.computation_mlir(w, *states) before any sync: the custom call, every copy, and the aliases.
     Saved in full to defer_update.mlir.

    TORCH_TPU_DEFER_AND_FUSE=1 PYTHONPATH=src python benchmarks/probe_defer.py
"""
import os, re, time, torch
from forge_tpu import FusedLinear, _bridge

V, H, BT, N, LR = 8192, 2048, 512, 20, 1e-4
DEFER = os.environ.get("TORCH_TPU_DEFER_AND_FUSE") == "1"


def sync(): torch.accelerator.synchronize()

def mib(): return torch.accelerator.memory_allocated() / 2**20

def layer():
    torch.manual_seed(0)
    fused = FusedLinear(H, V, lr = LR).to("tpu").train()
    fused.fusedStateObject.ensure_buffers(fused.weight)
    return fused

def timed(update, n):
    update(); sync()                                                                                     # first call: compiles, checks, keeps the launch
    held = mib(); torch.accelerator.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(n): update()
    issued = time.perf_counter(); sync(); end = time.perf_counter()
    return 1e3 * (end - start) / n, 1e3 * (issued - start) / n, torch.accelerator.max_memory_allocated() / 2**20 - held


if __name__ == "__main__":
    print(f"torch {torch.__version__}, TORCH_TPU_DEFER_AND_FUSE={'1' if DEFER else 'not set'}, w ({V}, {H}) bf16, go ({BT}, {V}), {N} updates")
    go = torch.randn(BT, V, generator = torch.Generator().manual_seed(1)).to("tpu", torch.bfloat16)
    x = torch.randn(BT, H, generator = torch.Generator().manual_seed(2)).to("tpu", torch.bfloat16)

    fused = layer()
    st = fused.fusedStateObject
    def forge_update():
        with torch.no_grad(): st.increment_step(); _bridge.fused_update(go, x, st, False)

    weight = torch.nn.Parameter(fused.weight.detach().clone())
    adamw = torch.optim.AdamW([weight], lr = LR, fused = True)
    def adamw_update():
        with torch.no_grad(): weight.grad = go.t() @ x
        adamw.step()

    print(f"\n{'update':<12} {'TPU ms/update':>14} {'host ms/update':>15} {'peak above held (MiB)':>22}")
    for name, update in (("FORGE", forge_update), ("fused AdamW", adamw_update)):
        ms, host, extra = timed(update, N)
        print(f"{name:<12} {ms:14.3f} {host:15.3f} {extra:22.1f}", flush = True)

    if DEFER:
        from torch_tpu._internal import sync as tpu_sync
        sync(); forge_update()                                                                           # one update, deferred: not run yet
        text = tpu_sync.computation_mlir([st.weight, *[t for a in st.buffers.values() for t in a.values()]])
        sync()
        with open("defer_update.mlir", "w") as f: f.write(text)
        lines = text.splitlines()
        print(f"\nDeferred program of one FORGE update: {len(lines)} lines, saved to defer_update.mlir")
        for label, pattern in (("custom calls", r"custom_call"), ("copies", r"\bcopy\b|stablehlo\.copy|mhlo\.copy"),
                               ("aliases / donation", r"alias|donat"), ("main signature", r"func\.func.*@main")):
            found = [l.strip() for l in lines if re.search(pattern, l)]
            print(f"  {label}: {len(found)}")
            for l in found[:6]: print(f"    {l[:300]}")
