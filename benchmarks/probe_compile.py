"""Which writes made inside a backward survive torch.compile(backend="tpu")?  One short run, one table.

Each probe trains a tiny layer for 2 steps, eager and compiled, and compares the tensors that the backward writes.
KEPT = compiled equals eager. LOST = compiled did not change them. Run on the TPU:

    PYTHONPATH=src python benchmarks/probe_compile.py

On a CPU (backend aot_eager) only probes 1-3 run: they check this script, not torch_tpu.
"""
import torch, torch.nn as nn

ON_TPU = torch.accelerator.is_available() and torch.accelerator.current_accelerator().type == "tpu"
DEV, BACKEND = ("tpu", "tpu") if ON_TPU else ("cpu", "aot_eager")
BT, H, V, LR = 64, 256, 256, 1e-2


# 1. plain torch writes in backward, into forward inputs (w, and m given to apply as a tensor)
class PlainFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, m):
        ctx.save_for_backward(x, w, m)
        return x @ w.t()
    @staticmethod
    def backward(ctx, go):
        x, w, m = ctx.saved_tensors
        dx = go @ w
        with torch.no_grad():
            m.mul_(0.9).add_(go.t() @ x); w.sub_(LR * m)
        return dx, None, None


# 2. a buffer written in the forward (as BatchNorm does with its running stats)
# 3. a custom_op that declares its writes (mutates_args), plain torch inside
@torch.library.custom_op("probe::torch_update", mutates_args = ("w", "m"))
def torch_update(go: torch.Tensor, x: torch.Tensor, w: torch.Tensor, m: torch.Tensor) -> None:
    m.mul_(0.9).add_(go.t() @ x); w.sub_(LR * m)

class OpFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, m):
        ctx.save_for_backward(x, w, m)
        return x @ w.t()
    @staticmethod
    def backward(ctx, go):
        x, w, m = ctx.saved_tensors
        dx = go @ w
        torch_update(go, x, w.detach(), m)
        return dx, None, None


# 4. the same custom_op design, with FORGE's Pallas kernel inside (jax_op, w and states donated, then copy_ back)
def forge_update_impl(go: torch.Tensor, x: torch.Tensor, w: torch.Tensor, states: list[torch.Tensor]) -> None:
    from forge_tpu import _bridge, kernel
    s = torch.tensor(kernel.OPTIMIZERS["adamw"].scalars({**kernel.OPTIMIZERS["adamw"].defaults, "lr": LR}, 1), dtype = torch.float32, device = w.device)
    out = _bridge._run_jax_op("adamw", "bf16")(s, [go, x], w, states)
    for old, new in zip([w, *states], out): old.copy_(new)

forge_update = torch.library.custom_op("probe::forge_update", forge_update_impl, mutates_args = ("w", "states"))

class ForgeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, m, v):
        ctx.save_for_backward(x, w, m, v)
        return x @ w.t()
    @staticmethod
    def backward(ctx, go):
        x, w, m, v = ctx.saved_tensors
        dx = go @ w
        forge_update(go.contiguous(), x.contiguous(), w.detach(), [m, v])
        return dx, None, None, None


class Layer(nn.Module):
    def __init__(self, probe):
        super().__init__()
        self.probe = probe
        self.weight = nn.Parameter(torch.randn(V, H, generator = torch.Generator().manual_seed(0)) / H**0.5)
        self.register_buffer("m", torch.zeros(V, H, dtype = torch.bfloat16 if probe == 4 else torch.float32))
        self.register_buffer("v", torch.zeros(V, H, dtype = torch.bfloat16))
        self.register_buffer("count", torch.zeros(()))

    def forward(self, x):
        if self.probe == 1: return PlainFn.apply(x, self.weight, self.m)
        if self.probe == 2: self.count.add_(1); return x @ self.weight.t()
        if self.probe == 3: return OpFn.apply(x, self.weight, self.m)
        if self.probe == 4: return ForgeFn.apply(x, self.weight, self.m, self.v)
        return self.fused(x)                                                                             # probe 5


def make(probe):
    layer = Layer(probe)
    if probe == 5:                                                                                        # 5. today's FusedLinear, unchanged
        from forge_tpu import FusedLinear
        layer.fused = FusedLinear(H, V, lr = LR)
        with torch.no_grad(): layer.fused.weight.copy_(layer.weight.to(layer.fused.weight.dtype))
    layer = layer.to(DEV)
    if probe == 5: layer.fused.fusedStateObject.ensure_buffers(layer.fused.weight)
    return layer


def written(layer):
    # the tensors the probe's update writes, on the host
    if layer.probe == 2: return [layer.count]
    if layer.probe == 5: return [layer.fused.weight, *[t for a in layer.fused.fusedStateObject.buffers.values() for t in a.values()]]
    return [layer.weight, layer.m] + ([layer.v] if layer.probe == 4 else [])


def train(probe, compiled):
    torch._dynamo.reset()
    layer = make(probe)
    before = [t.detach().float().cpu().clone() for t in written(layer)]
    ptrs = [t.data_ptr() for t in written(layer)]
    run = torch.compile(layer, backend = BACKEND, fullgraph = True) if compiled else layer
    x = torch.randn(BT, H, generator = torch.Generator().manual_seed(1)).to(DEV).requires_grad_()
    for _ in range(2): run(x).float().sum().backward()
    if ON_TPU: torch.accelerator.synchronize()
    after = [t.detach().float().cpu() for t in written(layer)]
    moved = max((a - b).abs().max().item() for a, b in zip(after, before))
    same_ptr = all(t.data_ptr() == p for t, p in zip(written(layer), ptrs))
    return after, moved, same_ptr


NAMES = {1: "plain torch writes in backward", 2: "buffer write in forward", 3: "custom_op(mutates_args), torch inside",
         4: "custom_op(mutates_args), FORGE jax_op inside", 5: "FusedLinear today (src unchanged)"}

if __name__ == "__main__":
    print(f"device {DEV}, backend {BACKEND}, torch {torch.__version__}")
    print(f"{'probe':<48} {'eager moved':>12} {'compiled moved':>15} {'max |compiled - eager|':>23} {'same data_ptr':>14}  verdict")
    for probe in (1, 2, 3, 4, 5) if ON_TPU else (1, 2, 3):
        try:
            eager, eager_moved, _ = train(probe, False)
            comp, comp_moved, same_ptr = train(probe, True)
            diff = max((a - b).abs().max().item() for a, b in zip(comp, eager))
            verdict = "LOST" if comp_moved == 0 else "KEPT" if diff <= 1e-2 * max(eager_moved, 1e-12) else "WRONG VALUES"
            print(f"{probe}. {NAMES[probe]:<45} {eager_moved:12.3e} {comp_moved:15.3e} {diff:23.3e} {str(same_ptr):>14}  {verdict}")
        except Exception as error:
            print(f"{probe}. {NAMES[probe]:<45} ERROR {type(error).__name__}: {str(error).splitlines()[0][:200]}")
