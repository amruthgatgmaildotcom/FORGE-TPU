"""Can FORGE's donating launch stop breaking pending readers of w?  Two tables (TPU only).

1. Which form of torch_tpu._internal.sync.synchronize starts ALL pending work without waiting: for each candidate call, a
   large pending matmul is made, the call is timed, and is_materializing / is_materialized tell whether it was started.
2. The two hazards tests.py found, for each setting of _bridge (KICK_EVERY, FLUSH_BEFORE_LAUNCH):
     pending clone     a clone() of w made before the step, not yet run, must still hold the old w after it
     pending forward   y = layer(x) with a loss that does not need y's value: y stays pending, and must still be readable
   "ok" = the value is right and nothing raised.

    TORCH_TPU_DEFER_AND_FUSE=1 PYTHONPATH=src python benchmarks/probe_donate.py
"""
import os, time, torch
from forge_tpu import FusedLinear, _bridge
from torch_tpu._internal import sync

H, V, BT = 2048, 8192, 512


def candidates():
    return [("synchronize(wait=False)", lambda: sync.synchronize(wait = False)), ("synchronize(False)", lambda: sync.synchronize(False)),
            ("synchronize([], False)", lambda: sync.synchronize([], False))]

def call_forms():
    print(f"{'call':<26} {'result':<56} {'host us':>8}")
    working = None
    for name, call in candidates():
        a = torch.randn(4096, 4096, device = "tpu"); torch.accelerator.synchronize()
        z = a @ a                                                                                        # pending (not started) with defer and fuse
        before = (sync.is_materializing(z), sync.is_materialized(z))
        try:
            t = time.perf_counter(); call(); us = 1e6 * (time.perf_counter() - t)
            after = (sync.is_materializing(z), sync.is_materialized(z))
            started = any(after)
            print(f"{name:<26} {f'pending matmul started: {started} (before {before}, after {after})':<56} {us:8.1f}")
            if started and working is None: working = (name, call)
        except Exception as error:
            print(f"{name:<26} {f'{type(error).__name__}: {str(error).splitlines()[0][:44]}':<56} {'':>8}")
        torch.accelerator.synchronize()
    return working

def layer():
    torch.manual_seed(0)
    fused = FusedLinear(H, V, lr = 1e-3).to("tpu").train()
    fused.fusedStateObject.ensure_buffers(fused.weight)
    return fused

def data():
    g = torch.Generator().manual_seed(1)
    return torch.randn(BT, H, generator = g).to("tpu", torch.bfloat16), torch.randn(BT, V, generator = g).to("tpu", torch.bfloat16)

def train_step(fused, x, dy):
    y = fused(x.clone().requires_grad_())
    (y * dy).sum().backward()                                                                            # dL/dy = dy: y's value is not needed
    return y

def hazard(kind):
    fused = layer(); x, dy = data()
    train_step(fused, x, dy); torch.accelerator.synchronize()                                            # first update: compiles, checks, keeps the launch
    try:
        if kind == "pending clone":
            old = fused.weight.detach().float().cpu()
            clone = fused.weight.detach().clone()
            train_step(fused, x, dy); torch.accelerator.synchronize()
            return "ok" if torch.equal(clone.float().cpu(), old) else "WRONG VALUE"
        w_old = fused.weight.detach().float().cpu()
        y = train_step(fused, x, dy); torch.accelerator.synchronize()
        expect = x.float().cpu() @ w_old.t()
        err = ((y.float().cpu() - expect).norm() / expect.norm()).item()
        return "ok" if err < 1e-2 else f"WRONG VALUE (rel {err:.1e})"
    except Exception as error:
        return f"{type(error).__name__}: {str(error).splitlines()[0][:60]}"


if __name__ == "__main__":
    print(f"torch {torch.__version__}, TORCH_TPU_DEFER_AND_FUSE={os.environ.get('TORCH_TPU_DEFER_AND_FUSE', 'not set')}\n")
    working = call_forms()
    if working:
        print(f"\nusing {working[0]} for _bridge._flush")
        _bridge._flush = working[1]
    else:
        print("\nno call form started the pending work: _flush keeps synchronize(wait=False)")
    print(f"\n{'setting':<38} {'pending clone':<40} {'pending forward':<40}")
    for kick, flush in ((1, False), (1, True), (0, True)):
        _bridge.KICK_EVERY, _bridge.FLUSH_BEFORE_LAUNCH = kick, flush
        print(f"{f'KICK_EVERY={kick}, FLUSH_BEFORE_LAUNCH={flush}':<38} {hazard('pending clone'):<40} {hazard('pending forward'):<40}", flush = True)
