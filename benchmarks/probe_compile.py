"""Which writes made inside a backward survive torch.compile(backend="tpu")?  One short run, one table.

Each probe trains a tiny layer for 2 steps, eager and compiled, and compares the tensors that the backward writes.
KEPT = compiled equals eager. LOST = compiled did not change them. Each probe runs eager, compiled, and compiled with
compiled autograd (the backward traced at run time as its own graph). "extra MiB" is the step's peak above what the TPU
held before it (step 2). The inputs and the first weights are exact in bf16 and the loss is a sum, so every matmul is
exact at any TPU matmul precision: compiled and eager must agree to rounding. Each probe runs in its own process, so a
crash (torch_tpu aborts the process) stops only that probe. Run on the TPU:

    PYTHONPATH=src python benchmarks/probe_compile.py

On a CPU (backend aot_eager) only probes 1, 2, 3 and 6 run: they check this script, not torch_tpu.
"""
import contextlib, subprocess, sys, torch, torch.nn as nn

ON_TPU = torch.accelerator.is_available() and torch.accelerator.current_accelerator().type == "tpu"
DEV, BACKEND = ("tpu", "tpu") if ON_TPU else ("cpu", "aot_eager")
BT, H, V, LR, DEPTH = 64, 1024, 1024, 1e-2, 8


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


# 4. the same custom_op design, with FORGE's Pallas kernel inside (jax_op, w and states donated, then copy_ back).
#    The scalars come in as a tensor made outside the trace: TPU run 2 aborted on a torch.tensor(...) made inside it.
def forge_update_impl(s: torch.Tensor, go: torch.Tensor, x: torch.Tensor, w: torch.Tensor, states: list[torch.Tensor]) -> None:
    from forge_tpu import _bridge
    out = _bridge._run_jax_op("adamw", "bf16")(s, [go, x], w, states)
    for old, new in zip([w, *states], out): old.copy_(new)

forge_update = torch.library.custom_op("probe::forge_update", forge_update_impl, mutates_args = ("w", "states"))

class ForgeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, m, v, s):
        ctx.save_for_backward(x, w, m, v, s)
        return x @ w.t()
    @staticmethod
    def backward(ctx, go):
        x, w, m, v, s = ctx.saved_tensors
        dx = go @ w
        forge_update(s, go.contiguous(), x.contiguous(), w.detach(), [m, v])
        return dx, None, None, None, None


class Layer(nn.Module):
    def __init__(self, probe):
        super().__init__()
        self.probe = probe
        self.weight = nn.Parameter((torch.randn(V, H, generator = torch.Generator().manual_seed(0)) / H**0.5).bfloat16().float())
        self.register_buffer("m", torch.zeros(V, H, dtype = torch.bfloat16 if probe == 4 else torch.float32))
        self.register_buffer("v", torch.zeros(V, H, dtype = torch.bfloat16))
        self.register_buffer("count", torch.zeros(()))
        if probe == 4:
            from forge_tpu import kernel
            self.register_buffer("s", torch.tensor(kernel.OPTIMIZERS["adamw"].scalars({**kernel.OPTIMIZERS["adamw"].defaults, "lr": LR}, 1), dtype = torch.float32))

    def forward(self, x):
        if self.probe == 1: return PlainFn.apply(x, self.weight, self.m)
        if self.probe == 2: self.count.add_(1); return x @ self.weight.t()
        if self.probe == 3: return OpFn.apply(x, self.weight, self.m)
        if self.probe == 4: return ForgeFn.apply(x, self.weight, self.m, self.v, self.s)
        if self.probe == 6:
            for layer in self.stack: x = layer(x)
            return x
        return self.fused(x.to(self.fused.weight.dtype))                                                 # probe 5


def make(probe):
    layer = Layer(probe)
    if probe == 5:                                                                                        # 5. today's FusedLinear, unchanged
        from forge_tpu import FusedLinear
        layer.fused = FusedLinear(H, V, lr = LR)
        with torch.no_grad(): layer.fused.weight.copy_(layer.weight.to(layer.fused.weight.dtype))
    if probe == 6: layer.stack = nn.ModuleList(Layer(1) for _ in range(DEPTH))                       # 6. probe 1, DEPTH layers deep
    layer = layer.to(DEV)
    if probe == 5: layer.fused.fusedStateObject.ensure_buffers(layer.fused.weight)
    return layer


def written(layer):
    # the tensors the probe's update writes, on the host
    if layer.probe == 2: return [layer.count]
    if layer.probe == 5: return [layer.fused.weight, *[t for a in layer.fused.fusedStateObject.buffers.values() for t in a.values()]]
    if layer.probe == 6: return [t for sub in layer.stack for t in written(sub)]
    return [layer.weight, layer.m] + ([layer.v] if layer.probe == 4 else [])


def train(probe, mode):
    torch._dynamo.reset()
    layer = make(probe)
    before = [t.detach().float().cpu().clone() for t in written(layer)]
    ptrs = [t.data_ptr() for t in written(layer)]
    run = layer if mode == "eager" else torch.compile(layer, backend = BACKEND, fullgraph = True)
    backward_ctx = lambda: torch._dynamo.compiled_autograd._enable(torch.compile(backend = BACKEND, fullgraph = True)) if mode == "compiled autograd" else contextlib.nullcontext()
    x = torch.randn(BT, H, generator = torch.Generator().manual_seed(1)).bfloat16().float().to(DEV).requires_grad_()
    extra = 0.0
    for _ in range(2):
        if ON_TPU: torch.accelerator.synchronize(); held = torch.accelerator.memory_allocated(); torch.accelerator.reset_peak_memory_stats()
        with backward_ctx(): run(x).float().sum().backward()
        x.grad = None
        if ON_TPU: torch.accelerator.synchronize(); extra = (torch.accelerator.max_memory_allocated() - held) / 2**20
    after = [t.detach().float().cpu() for t in written(layer)]
    moved = max((a - b).abs().max().item() for a, b in zip(after, before))
    same_ptr = all(t.data_ptr() == p for t, p in zip(written(layer), ptrs))
    return after, moved, same_ptr, extra


NAMES = {1: "plain torch writes in backward", 2: "buffer write in forward", 3: "custom_op(mutates_args), torch inside",
         4: "custom_op(mutates_args), FORGE jax_op inside", 5: "FusedLinear today (src unchanged)",
         6: f"probe 1, {DEPTH} layers deep (memory; values 1e-2)"}

def run_probe(probe):
    eager = None
    for mode in ("eager", "compiled", "compiled autograd"):
        try:
            out, moved, same_ptr, extra = train(probe, mode)
            if mode == "eager": eager, eager_moved, verdict, diff = out, moved, "reference", 0.0
            else:
                diff = max((a - b).abs().max().item() for a, b in zip(out, eager))
                tolerance = 1e-2 if probe == 6 else 1e-4                                                 # 6: step 2's lower layers get go through updated (not bf16-exact) weights
                verdict = "LOST" if moved == 0 else "KEPT" if diff <= tolerance * max(eager_moved, 1e-12) else "WRONG VALUES"
            print(f"{probe}. {NAMES[probe]:<45} {mode:<18} {moved:10.3e} {diff:16.3e} {str(same_ptr):>14} {extra:10.1f}  {verdict}", flush = True)
        except Exception as error:
            print(f"{probe}. {NAMES[probe]:<45} {mode:<18} ERROR {type(error).__name__}: {str(error).splitlines()[0][:200]}", flush = True)


if __name__ == "__main__":
    if sys.argv[1:]:                                                                                     # one probe, in this process
        run_probe(int(sys.argv[1]))
    else:                                                                                                # every probe, each in its own process
        print(f"device {DEV}, backend {BACKEND}, torch {torch.__version__}")
        print(f"{'probe':<48} {'mode':<18} {'moved':>10} {'max |x - eager|':>16} {'same data_ptr':>14} {'extra MiB':>10}  verdict", flush = True)
        for probe in (1, 2, 3, 4, 5, 6) if ON_TPU else (1, 2, 3, 6):
            child = subprocess.run([sys.executable, "-u", __file__, str(probe)], capture_output = True, text = True)
            print(child.stdout, end = "", flush = True)
            if child.returncode:
                why = [l.strip() for l in child.stderr.splitlines() if "Check failed" in l or "Error" in l][:1] or [child.stderr.strip()[-200:]]
                print(f"{probe}. {NAMES[probe]:<45} CRASHED (exit code {child.returncode}): {why[0][:300]}", flush = True)
