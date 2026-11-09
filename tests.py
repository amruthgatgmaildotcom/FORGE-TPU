"""FORGE-TPU correctness, against official PyTorch. Run on the TPU (CPU runs the same checks through Pallas' interpreter,
slowly, with small shapes only, and skips the TPU-only checks):

    PYTHONPATH=src python tests.py                          eager (plain)
    TORCH_TPU_DEFER_AND_FUSE=1 PYTHONPATH=src python tests.py   eager, defer and fuse (as the benchmark)

The reference is torch.optim.AdamW(fused=True) on an nn.Linear that starts from the same weights, gets the same input and
the same output gradient (loss = (y * g).sum(), so dL/dy = g exactly), with the same hyperparameters. torch keeps its states
in the weight's dtype, so bf16 weights compare FORGE bf16 states with torch bf16 states (same storage). fp32 weights have no
torch equivalent for bf16 states (torch's are fp32): they must still be close (same limits). int8 states have no torch
equivalent at all: they run side by side and the distance is reported (INFO), as in dk4248/FORGE.

Error measure: rel(a, b) = ||a - b|| / ||b|| over the whole tensor. For the weight it is taken on the change (w_now - w_start),
so it measures the update, not the (much larger) weight. Limits are printed with each result; they come from the expected
rounding (torch rounds dW to the weight dtype before AdamW, FORGE never materializes dW and uses it in fp32).
Prints one line per check, PASS / FAIL / SKIP / INFO and its numbers; exit code 1 if anything failed.
"""
import copy, math, os, sys, time, traceback, torch, torch.nn as nn, torch.nn.functional as F
from forge_tpu import FusedLinear

ON_TPU = torch.accelerator.is_available() and torch.accelerator.current_accelerator().type == "tpu"
DEV = "tpu" if ON_TPU else "cpu"
HP = dict(lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0.01)
FORGE_HP = dict(lr = HP["lr"], beta1 = HP["betas"][0], beta2 = HP["betas"][1], eps = HP["eps"], weight_decay = HP["weight_decay"])
STEPS = 5
SHAPES = [(256, 512, 128), (384, 640, 200)] + ([(2048, 8192, 512)] if ON_TPU else [])                    # (in H, out V, tokens BT)
RESULTS = []
torch.set_float32_matmul_precision("highest")                                                            # torch_tpu runs fp32 matmuls in bf16 by default
# (torch_tpu precision.py : "On TPU, float32 matrix multiplications are natively executed in bfloat16"); FORGE's kernel uses
# HIGHEST for fp32 (fused_grad.py), so the torch reference must too, or the comparison measures torch's rounding (TPU run 1)


# ---- helpers ----
def sync():
    if ON_TPU: torch.accelerator.synchronize()

def host(t):
    return t.detach().float().cpu().clone()                                                               # always a copy (on a CPU, .float().cpu() of fp32 is the same tensor)

def rel(a, b):
    a, b = host(a), host(b)
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()

def check(name, ok, detail):
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name:<58} {detail}", flush = True)

def info(name, detail):
    RESULTS.append((name, "INFO", detail))
    print(f"INFO  {name:<58} {detail}", flush = True)

def skip(name, why):
    RESULTS.append((name, "SKIP", why))
    print(f"SKIP  {name:<58} {why}", flush = True)

def pair(H, V, dtype, bias, state_mode = "bf16", seed = 0):
    # the torch reference (nn.Linear + fused AdamW) and FORGE, from the same weights
    torch.manual_seed(seed)
    host_ref = nn.Linear(H, V, bias = bias, dtype = dtype)                                               # made on the host, then moved twice: two buffers that
    ref = copy.deepcopy(host_ref).to(DEV)                                                                 # share nothing (a deepcopy on the TPU can share w's buffer
    fused = FusedLinear.from_linear(host_ref, state_mode = state_mode, **FORGE_HP).to(DEV).train()       # with ref until a write: TPU runs 1-2, see pending_clone)
    ref_opt = torch.optim.AdamW(ref.parameters(), **HP, fused = True)
    bias_opt = torch.optim.AdamW([fused.bias], **HP, fused = True) if bias else None                   # FORGE leaves the bias to an optimizer
    sync()
    return ref, ref_opt, fused, bias_opt

def data(BT, H, V, dtype, lead = (), seed = 1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*lead, BT, H, generator = g).to(DEV, dtype)
    dy = (torch.randn(*lead, BT, V, generator = g) / math.sqrt(BT)).to(DEV, dtype)
    return x, dy

def step(layer, opt, x, dy, x_grad = True):
    xin = x.clone().requires_grad_(x_grad)
    y = layer(xin)
    (y * dy).sum().backward()
    if opt: opt.step(); opt.zero_grad(set_to_none = True)
    return y, xin.grad

def forge_states(fused):
    # FORGE's m and v, decoded to fp32 tensors (int8: q * scale per 64 rows)
    st, out = fused.fusedStateObject, []
    for name in ("m", "v"):
        arrays = st.buffers[name]
        if "data" in arrays: out.append(arrays["data"].float())
        else:
            q, scale = arrays["q"].float(), arrays["scale"].float()
            out.append((q.reshape(q.shape[0] // 64, 64, q.shape[1]) * scale[:, None, :]).reshape(q.shape))
    return out


# ---- 1. the update against torch AdamW ----
def against_torch(H, V, BT, dtype, bias, state_mode, lead = (), limit_w = 3e-2, limit_state = 3e-2):
    tag = f"{state_mode} states, {str(dtype)[6:]} w, H{H} V{V} BT{BT}{', bias' if bias else ''}{', 3-D input' if lead else ''}"
    ref, ref_opt, fused, bias_opt = pair(H, V, dtype, bias, state_mode)
    x, dy = data(BT, H, V, dtype, lead)
    w0, b0 = host(ref.weight), host(ref.bias) if bias else None
    for i in range(STEPS):
        w_before = host(fused.weight)
        y_ref, dx_ref = step(ref, ref_opt, x, dy)
        y_f, dx_f = step(fused, bias_opt, x, dy)
        if i == 0: first_y, first_dx = rel(y_f, y_ref), rel(dx_f, dx_ref)                              # step 1 only: both still have the same weights
    sync()
    last_step = (host(fused.weight) - w_before).abs().max().item() / HP["lr"]                             # the largest element step of the last update, in lr
    dw = rel(host(fused.weight) - w0, host(ref.weight) - w0)
    m_f, v_f = forge_states(fused)
    st_ref = ref_opt.state[ref.weight]
    dm, dv = rel(m_f, st_ref["exp_avg"]), rel(v_f, st_ref["exp_avg_sq"])
    db = rel(host(fused.bias) - b0, host(ref.bias) - b0) if bias else 0.0
    numbers = f"rel: dW {dw:.2e} m {dm:.2e} v {dv:.2e}, step 1: y {first_y:.1e} dx {first_dx:.1e}, bias {db:.1e}, last step max |dw| {last_step:.1f} lr"
    if state_mode == "int8":                                                                              # no torch equivalent: reported, as in dk4248/FORGE
        return info(f"int8 side by side: {tag}", numbers + " (torch: <= ~1 lr)")
    ok = first_y < 1e-2 and first_dx < 1e-2 and fused.fusedStateObject.step == STEPS and fused.weight.grad is None and db < 1e-2
    ok = ok and dw < limit_w and dm < limit_state and dv < limit_state                                    # bf16 states: torch's (bf16 w) or close to them (fp32 w)
    check(f"vs torch AdamW: {tag}", ok, numbers + f" (limits: y, dx, bias 1e-2; dW {limit_w:.0e}; m, v {limit_state:.0e})")


# ---- 2. facilities ----
def eval_mode():
    ref, _, fused, _ = pair(256, 512, torch.bfloat16, True)
    x, _ = data(64, 256, 512, torch.bfloat16)
    fused.eval()
    w0 = host(fused.weight)
    y = fused(x.requires_grad_())
    y.sum().backward()
    sync()
    same = torch.equal(host(y), host(F.linear(x.detach(), ref.weight, ref.bias)))
    check("eval(): plain F.linear, no update, no state", same and torch.equal(host(fused.weight), w0) and fused.fusedStateObject.buffers is None
          and fused.weight.grad is not None, f"output equal {same}, weight unchanged, buffers {fused.fusedStateObject.buffers}, .grad as nn.Linear")

def from_linear_shares():
    linear = nn.Linear(256, 512, bias = True, dtype = torch.bfloat16)
    fused = FusedLinear.from_linear(linear)
    check("from_linear(): reuses the same Parameters (no copy)", fused.weight is linear.weight and fused.bias is linear.bias, "identity of weight and bias")

def no_input_grad():
    ref, ref_opt, fused, _ = pair(256, 512, torch.bfloat16, False)
    x, dy = data(64, 256, 512, torch.bfloat16)
    w0 = host(ref.weight)
    step(ref, ref_opt, x, dy, x_grad = False); step(fused, None, x, dy, x_grad = False)
    sync()
    dw = rel(host(fused.weight) - w0, host(ref.weight) - w0)
    check("input without grad (first layer): no dx, update still right", dw < 3e-2, f"rel dW {dw:.2e} (limit 3e-2)")

def state_dict_roundtrip():
    _, _, a, _ = pair(256, 512, torch.bfloat16, False)
    x, dy = data(64, 256, 512, torch.bfloat16)
    for _ in range(3): step(a, None, x, dy)
    saved, w_saved = a.fusedStateObject.state_dict(), host(a.weight)
    for _ in range(2): step(a, None, x, dy)
    b = FusedLinear(256, 512, dtype = torch.bfloat16, **FORGE_HP).to(DEV).train()
    with torch.no_grad(): b.weight.copy_(w_saved.to(DEV, torch.bfloat16))
    b.fusedStateObject.ensure_buffers(b.weight)
    b.fusedStateObject.load_state_dict(saved)
    for _ in range(2): step(b, None, x, dy)
    sync()
    same = torch.equal(host(a.weight), host(b.weight)) and all(torch.equal(host(s), host(t)) for s, t in zip(forge_states(a), forge_states(b)))
    check("state_dict() / load_state_dict(): resume = uninterrupted", same and b.fusedStateObject.step == 5, f"bit for bit {same}, step {b.fusedStateObject.step}")

def int8_shape_check():
    try:
        FusedLinear(256, 100, state_mode = "int8").to(DEV).train()(torch.zeros(8, 256, dtype = torch.bfloat16, device = DEV))
        check("int8 with out_features % 64 != 0 raises ValueError", False, "no error raised")
    except ValueError as error:
        check("int8 with out_features % 64 != 0 raises ValueError", True, str(error)[:70])

def bad_dtype():
    try:
        FusedLinear(256, 512, dtype = torch.float16)
        check("weight dtype other than fp32 / bf16 raises ValueError", False, "no error raised")
    except ValueError as error:
        check("weight dtype other than fp32 / bf16 raises ValueError", True, str(error)[:70])


# ---- 3. TPU only: donation hazards, memory, torch.compile ----
def pending_clone_torch_only():
    # the same hazard without FORGE: a pending clone of a tensor that an in-place op then updates (torch_tpu donates in-place buffers)
    if not ON_TPU: return skip("torch only: a pending clone() survives an in-place add_", "TPU only")
    a = torch.randn(2048, 2048, generator = torch.Generator().manual_seed(2)).to(DEV)
    sync()
    before = host(a)
    clone = a.clone()                                                                                     # issued, not run yet
    try:
        a.add_(1); sync()
        same = torch.equal(host(clone), before)
        check("torch only: a pending clone() survives an in-place add_", same, f"clone equals the old value: {same} (fails = torch_tpu, not FORGE)")
    except Exception as error:
        check("torch only: a pending clone() survives an in-place add_", False, f"{type(error).__name__}: {str(error).splitlines()[0][:90]} (torch_tpu, not FORGE)")

def pending_clone():
    # a clone of w made before the update, not yet run, must still hold the old w after FORGE donated w's buffer
    if not ON_TPU: return skip("a pending clone() of w survives the update (donation)", "TPU only")
    _, _, fused, _ = pair(2048, 8192, torch.bfloat16, False)
    x, dy = data(512, 2048, 8192, torch.bfloat16)
    sync()
    before = host(fused.weight)
    clone = fused.weight.detach().clone()                                                                 # issued, not run yet
    from torch_tpu._internal import sync as tpu_sync
    how = (f"copy-on-write w/clone {torch._C._is_cow_tensor(fused.weight)}/{torch._C._is_cow_tensor(clone)}, clone materializing/materialized "
           f"{tpu_sync.is_materializing(clone)}/{tpu_sync.is_materialized(clone)}")                      # how torch_tpu holds the clone before the update
    try:
        step(fused, None, x, dy); sync()
        same = torch.equal(host(clone), before)
        check("a pending clone() of w survives the update (donation)", same, f"clone equals the old w: {same}; {how}")
    except Exception as error:
        check("a pending clone() of w survives the update (donation)", False, f"{type(error).__name__}: {str(error).splitlines()[0][:60]}; {how}")

def memory_without_sync():
    # N steps without a sync must not hold more memory than N steps with one (a sync after each step)
    if not ON_TPU: return skip("memory: 20 steps without sync vs with sync", "TPU only")
    peaks = []
    for synced in (True, False):
        _, _, fused, _ = pair(2048, 8192, torch.bfloat16, False)
        x, dy = data(512, 2048, 8192, torch.bfloat16)
        step(fused, None, x, dy); sync()
        held = torch.accelerator.memory_allocated(); torch.accelerator.reset_peak_memory_stats()
        for _ in range(20):
            step(fused, None, x, dy)
            if synced: sync()
        sync(); peaks.append((torch.accelerator.max_memory_allocated() - held) / 2**20)
        del fused
    layer = 3 * 8192 * 2048 * 2 / 2**20                                                                   # w + m + v
    check("memory: 20 steps without sync vs with sync", peaks[1] <= peaks[0] + layer / 2,
          f"peak above held: synced {peaks[0]:.0f} MiB, not synced {peaks[1]:.0f} MiB (one layer's w+m+v = {layer:.0f} MiB)")

def compiled():
    # torch.compile(backend="tpu") must give the same weights as eager
    if not ON_TPU: return skip("torch.compile(backend='tpu') = eager", "TPU only")
    outs = []
    for compile_it in (False, True):
        torch._dynamo.reset()
        _, _, fused, _ = pair(256, 512, torch.bfloat16, False)
        x, dy = data(64, 256, 512, torch.bfloat16)
        w0 = host(fused.weight)
        run = torch.compile(fused, backend = "tpu", fullgraph = True) if compile_it else fused
        try:
            for _ in range(STEPS): step(run, None, x, dy)
            sync(); outs.append(host(fused.weight) - w0)
        except Exception as error:
            return check("torch.compile(backend='tpu') = eager", False, f"{type(error).__name__}: {str(error).splitlines()[0][:90]}")
    moved = outs[1].abs().max().item()
    check("torch.compile(backend='tpu') = eager", moved > 0 and rel(outs[1], outs[0]) < 1e-2,
          f"compiled moved w by {moved:.2e}, rel to eager {rel(outs[1], outs[0]):.2e} (known: torch_tpu drops writes in a compiled backward)")


if __name__ == "__main__":
    mode = "torch.compile n/a" if not ON_TPU else f"TORCH_TPU_DEFER_AND_FUSE={os.environ.get('TORCH_TPU_DEFER_AND_FUSE', 'not set')}"
    print(f"FORGE-TPU tests on {DEV}, torch {torch.__version__}, {mode}, {STEPS} steps per comparison\n", flush = True)
    started = time.perf_counter()
    cases = [lambda H = H, V = V, BT = BT, d = d, b = b, s = s: against_torch(H, V, BT, d, b, s)
             for (H, V, BT) in SHAPES for d in (torch.bfloat16, torch.float32) for s in ("bf16", "int8") for b in (False,)]
    cases += [lambda: against_torch(256, 512, 128, torch.bfloat16, True, "bf16"), lambda: against_torch(256, 512, 64, torch.bfloat16, False, "bf16", lead = (2,))]
    cases += [eval_mode, from_linear_shares, no_input_grad, state_dict_roundtrip, int8_shape_check, bad_dtype, pending_clone_torch_only, pending_clone, memory_without_sync, compiled]
    for case in cases:
        try: case()
        except Exception as error:
            RESULTS.append((getattr(case, "__name__", "case"), "FAIL", repr(error)))
            print(f"FAIL  {getattr(case, '__name__', 'case'):<58} crashed: {type(error).__name__}: {str(error).splitlines()[0][:120]}", flush = True)
            traceback.print_exc(limit = 3)
    counts = {k: sum(r[1] == k for r in RESULTS) for k in ("PASS", "FAIL", "SKIP", "INFO")}
    print(f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped, {counts['INFO']} reported in {time.perf_counter() - started:.0f} s")
    sys.exit(1 if counts["FAIL"] else 0)
