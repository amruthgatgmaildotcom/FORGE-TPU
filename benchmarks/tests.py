import gc, math, os, re, torch, torch.nn as nn, torch.nn.functional as F
from forge_tpu import FusedLinear
import llama as L
from methods import OPTIM, OPTIMIZERS                                 # the optimizer of this run (OPTIM) : adamw against torch.optim.AdamW, sgd against torch.optim.SGD

# FORGE correctness : every check compares FORGE with PyTorch. The reference is PyTorch on CPU in fp32 / fp64 (assumed correct). One optimizer per run (OPTIM).
# SGD (no momentum) has no state : the checks of m and v and the int8 cases are AdamW only.
# As in FORGE (dk4248) tests/test_correctness.py : one AdamW step from a non-zero state (step 5), the same (BT, V, H) shapes, the same tolerances.
# Added : bf16-stored values must be at most 1 bf16 ulp from the rounded reference, real-model gradient sizes, several steps through two layers
# (the TPU replay path and the dx order), and a small Llama trained with all four methods against the fp32 reference.
DEVICE = "tpu"
TOL, QTOL, ULP, BASE = 5e-3, 5e-2, 1.0, 2.0     # fp32 paths, int8 state (as in FORGE tests) | bf16-stored values, in ulps | model : FORGE error <= BASE x PyTorch bf16 error
LR, B1, B2, EPS, WD = 1e-3, 0.9, 0.999, 1e-8, 0.01
SHAPES = [(128, 256, 128), (64, 512, 256), (256, 1024, 512), (1500, 1536, 2048)]   # (BT, V, H) : the 3 FORGE test shapes + a multi-tile one with a partial BT tile
OUT = os.environ.get("RESULTS", "results")
RESULTS = []                                                           # (case, check, error, tolerance)
NA = []                                                                # (case, why) : a method that could not run (e.g. fused SGD not supported on this TPU)


#------------------------------------------------------------------------- tools ---------------------------------------------------------------------------#

def rel(a, b):                                                         # max |a - b| / max |b|, in fp64 on CPU
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-300)).item()

def ulps(a, ref):
    # max distance of a from bf16(ref), in bf16 ulps of ref. The ulp has a floor of 1e-6 x max|ref| (far below bf16's 4e-3) : near zero, after
    # cancellation, the normal fp32 summation noise of the gradient is many ulps of a tiny value
    r = ref.detach().double().cpu().to(torch.bfloat16).double()
    ulp = torch.pow(2.0, torch.frexp(r)[1] - 8.0).clamp_min(1e-6 * r.abs().max().item())
    return ((a.detach().double().cpu() - r).abs() / ulp).max().item()

def case(tag, items):                                                  # items : [(check, error, tolerance)]
    RESULTS.extend((tag, *item) for item in items)
    ok = all(e <= t for _, e, t in items)
    print(f"{'PASS' if ok else 'FAIL'}  {tag:52s} " + "  ".join(f"{c} {e:.2g}{'' if e <= t else ' (> ' + format(t, 'g') + ')'}" for c, e, t in items), flush = True)

def torch_adamw(w, g, m, v, step):
    # the reference : one torch.optim.AdamW step on CPU in fp32, from state (m, v) after step - 1 steps. Returns (w, m, v)
    p = nn.Parameter(w.detach().float().cpu().clone()); p.grad = g.float().cpu()
    opt = torch.optim.AdamW([p], lr = LR, betas = (B1, B2), eps = EPS, weight_decay = WD, foreach = False)
    opt.state[p] = {"step": torch.tensor(float(step - 1)), "exp_avg": m.float().cpu().clone(), "exp_avg_sq": v.float().cpu().clone()}
    opt.step()
    return p.detach(), opt.state[p]["exp_avg"], opt.state[p]["exp_avg_sq"]

def torch_sgd(w, g):
    # the reference for sgd : one torch.optim.SGD step on CPU in fp32 (no momentum, no state). Returns (w,)
    p = nn.Parameter(w.detach().float().cpu().clone()); p.grad = g.float().cpu()
    torch.optim.SGD([p], lr = LR, weight_decay = WD).step()
    return (p.detach(),)

def reference(w, g, m, v, step):                                       # the torch step of this run's optimizer
    return torch_adamw(w, g, m, v, step) if OPTIM == "adamw" else torch_sgd(w, g)

def forge_layer(w, mode, b = None):
    lin = nn.Linear(w.shape[1], w.shape[0], bias = b is not None, device = DEVICE, dtype = w.dtype)
    with torch.no_grad():
        lin.weight.copy_(w.to(DEVICE))
        if b is not None: lin.bias.copy_(b.to(DEVICE))
    kwargs = dict(state_mode = mode, beta1 = B1, beta2 = B2, eps = EPS) if OPTIM == "adamw" else dict(optimizer_type = "sgd")   # sgd : no state, no state_mode
    layer = FusedLinear.from_linear(lin, lr = LR, weight_decay = WD, **kwargs).train()
    layer.fusedStateObject.ensure_buffers(layer.weight)
    return layer

def get_state(layer):                                                  # (m, v) as fp32 on CPU, int8 decoded (no floor). sgd : (None, None)
    if OPTIM != "adamw": return None, None
    def read(a):
        if "data" in a: return a["data"].float().cpu()
        q, scale = a["q"].float().cpu(), a["scale"].float().cpu()
        return (q.view(q.shape[0] // 64, 64, -1) * scale[:, None, :]).view(q.shape)
    s = layer.fusedStateObject.buffers
    return read(s["m"]), read(s["v"])

def set_state(layer, m, v, step):                                      # state after step - 1 steps, stored as FORGE stores it
    state = layer.fusedStateObject
    state.step = step - 1
    if OPTIM != "adamw": return                                        # sgd : no state
    for a, t in ((state.buffers["m"], m), (state.buffers["v"], v)):
        if "data" in a: a["data"].copy_(t.to(DEVICE, a["data"].dtype)); continue
        blocks = t.view(t.shape[0] // 64, 64, -1)
        scale = blocks.abs().amax(1) / 127 + 1e-12
        a["q"].copy_((blocks / scale[:, None, :]).round().clamp(-127, 127).view(t.shape).to(DEVICE, torch.int8)); a["scale"].copy_(scale.to(DEVICE))

def update_checks(layer, w_old, ref, mode):
    # FORGE's new w, m, v against the reference. bf16 w : ulps of the new w. fp32 w : the update w_new - w_old (a check on w itself would not see a wrong update)
    w, (m, v) = layer.weight, get_state(layer)
    ew = ulps(w, ref[0]) if w.dtype == torch.bfloat16 else rel(w.float().cpu() - w_old.float(), ref[0] - w_old.float())
    unit = "ulp" if w.dtype == torch.bfloat16 else "rel"
    if OPTIM != "adamw": return [(f"w({unit})", ew, ULP if unit == "ulp" else TOL)]   # sgd : no state to check
    es = [rel(a, b) if mode == "int8" else ulps(a, b) for a, b in ((m, ref[1]), (v, ref[2]))]
    return [(f"w({unit})", ew, ULP if unit == "ulp" else TOL), ("m", es[0], QTOL if mode == "int8" else ULP), ("v", es[1], QTOL if mode == "int8" else ULP)]


#------------------------------------------------------------------------- checks --------------------------------------------------------------------------#

def module_checks():
    lin = nn.Linear(256, 128, bias = False, device = DEVICE, dtype = torch.bfloat16)
    layer = FusedLinear.from_linear(lin).eval()
    x = torch.randn(8, 256, dtype = torch.bfloat16).to(DEVICE)
    case("module : from_linear, eval", [("shares weight (0 = yes)", float(layer.weight is not lin.weight), 0), ("eval out = F.linear", rel(layer(x), F.linear(x, lin.weight)), 0)])


def layer_step(BT, V, H, wdtype, mode, scale = 1.0, bias = False):
    # one FusedLinear forward + backward (= one AdamW step, step 5, non-zero state) against F.linear, autograd math and torch.optim.AdamW
    g = torch.Generator().manual_seed(0)
    x, go = torch.randn(BT, H, generator = g).to(wdtype), (torch.randn(BT, V, generator = g) * scale).to(wdtype)
    w, b = (torch.randn(V, H, generator = g) / math.sqrt(H)).to(wdtype), (torch.randn(V, generator = g).to(wdtype) if bias else None)
    grad = go.double().T @ x.double()
    m, v = torch.zeros(V, H), torch.zeros(V, H)
    for _ in range(4):                                                 # a real state : AdamW after 4 random gradients of the same size
        gi = torch.randn(V, H, generator = g) * grad.std().item()
        m, v = B1 * m + (1 - B1) * gi, B2 * v + (1 - B2) * gi * gi
    layer = forge_layer(w, mode, b)
    set_state(layer, m, v, step = 5)
    m0, v0 = get_state(layer)                                          # what FORGE really holds : the reference starts from it
    xd = x.to(DEVICE).requires_grad_()
    y = layer(xd); y.backward(go.to(DEVICE))
    y_ref = x.double() @ w.double().T + (0 if b is None else b.double())
    items = [("y", rel(y, y_ref), TOL), ("dx", rel(xd.grad, go.double() @ w.double()), TOL)]
    if bias: items.append(("db", rel(layer.bias.grad, go.double().sum(0)), TOL))
    tag = f"step {OPTIM} {mode or 'no'} state, {str(wdtype)[6:]} w, {BT}x{V}x{H}" + (f", g x{scale:g}" if scale != 1 else "") + (", bias" if bias else "")
    case(tag, items + update_checks(layer, w, reference(w, grad, m0, v0, 5), mode))


def chain(mode, steps = 10, BT = 512, H = 1024):
    # y = l2(l1(x)), two same-shape bf16 layers, 10 steps. Each step is checked from FORGE's own state before it, so errors cannot add up.
    # Steps 2+ use the replayed TPU launch, and l1 gets l2's dx : this checks that dx is made with the OLD weight
    g = torch.Generator().manual_seed(1)
    l1, l2 = (forge_layer((torch.randn(H, H, generator = g) / math.sqrt(H)).to(torch.bfloat16), mode) for _ in range(2))
    worst = {}
    for step in range(1, steps + 1):
        before = [(l.weight.detach().cpu().clone(), *get_state(l)) for l in (l1, l2)]
        x = torch.randn(BT, H, generator = g).to(torch.bfloat16).to(DEVICE).requires_grad_()
        go = torch.randn(BT, H, generator = g).to(torch.bfloat16)
        h = l1(x); h.retain_grad(); l2(h).backward(go.to(DEVICE))
        (w1, m1, v1), (w2, m2, v2) = before
        hc, dh = h.detach().cpu().double(), h.grad.cpu().double()      # what the TPU really gave l2 and l1
        items = [("dh", rel(dh, go.double() @ w2.double()), TOL), ("dx", rel(x.grad, dh @ w1.double()), TOL)]
        items += [(f"l2 {c}", e, t) for c, e, t in update_checks(l2, w2, reference(w2, go.double().T @ hc, m2, v2, step), mode)]
        items += [(f"l1 {c}", e, t) for c, e, t in update_checks(l1, w1, reference(w1, dh.T @ x.detach().cpu().double(), m1, v1, step), mode)]
        for c, e, t in items: worst[c] = (max(e, worst.get(c, (0,))[0]), t)
    case(f"chain {OPTIM} {mode or 'no'} state, 2 layers, {steps} steps (worst)", [(c, e, t) for c, (e, t) in worst.items()])


def train(model, opt, data, steps):
    snap = lambda: {n: p.detach().float().cpu().clone() for n, p in model.named_parameters()}
    w0, losses = snap(), []
    for step in range(steps):
        x, y = data[step % len(data)]
        loss = L.loss_fn(model(x), y); loss.backward()
        opt.step(); opt.zero_grad(set_to_none = True); losses.append(loss.item())
        if step == 0: w1 = snap()
    return w0, w1, snap(), losses


def model_check(steps = 30):
    # a small Llama (llama.py's model : transformers' LlamaForCausalLM with small numbers) trained with every method of OPTIM on the TPU, against the torch optimizer on CPU in fp32 from the same init.
    # bf16 training is never exact, so FORGE passes if its error is at most BASE x the larger error of PyTorch's own bf16 methods (vanilla, fused : those that ran)
    L.V, L.D, L.FF, L.LAYERS, L.HEADS, L.KV_HEADS, L.SEQ, L.ACTIVE, L.DEVICE = 4096, 512, 1536, 2, 8, 2, 256, 256, DEVICE
    L.HD = L.D // L.HEADS
    data = L.make_data()
    torch.manual_seed(0); ref = L.Llama().to(torch.bfloat16).float()   # the same init as make_model, in fp32 on CPU
    ref_opt = (torch.optim.AdamW if OPTIM == "adamw" else torch.optim.SGD)(ref.parameters(), lr = L.LR, weight_decay = L.WD, foreach = False)
    R = {"reference": train(ref, ref_opt, [(x.cpu(), y.cpu()) for x, y in data], steps)}
    for name in OPTIMIZERS:
        try:
            model, opt = L.make_model(name); model.train()
            R[name] = train(model, opt, data, steps); del model, opt
        except Exception as error:                                     # e.g. fused SGD not supported on this TPU : NA, the other methods go on
            NA.append((f"model {name}", f"{type(error).__name__}: {str(error)[:150]}")); print(f"NA  model {name} : {NA[-1][1]}", flush = True)
        gc.collect()
    r0, r1, rf, rl = R.pop("reference")
    kinds = sorted({re.sub(r"layers\.\d+\.", "", n) for n in r0})
    def errors(name):                                                  # {metric : error} for one method
        w0, w1, wf, losses = R[name]
        e = {f"step 1 dw {k}": max(rel(w1[n] - w0[n], r1[n] - r0[n]) for n in r0 if re.sub(r"layers\.\d+\.", "", n) == k) for k in kinds}
        e |= {f"{steps} steps dw {k}": max(rel(wf[n] - w0[n], rf[n] - r0[n]) for n in r0 if re.sub(r"layers\.\d+\.", "", n) == k) for k in kinds}
        return e | {"max |loss - ref|": max(abs(a - b) for a, b in zip(losses, rl))}
    E = {name: errors(name) for name in R}
    torch_names, forge_names = [n for n in OPTIMIZERS if not n.startswith("forge") and n in E], [n for n in OPTIMIZERS if n.startswith("forge") and n in E]
    print(f"\nsmall Llama ({L.LAYERS} layers, D={L.D}, V={L.V}), {steps} steps, error against torch {OPTIM} fp32 on CPU :")
    print(f"{'':32s}" + "".join(f"{n:>13s}" for n in OPTIMIZERS))
    for metric in next(iter(E.values()), {}): print(f"{metric:32s}" + "".join(f"{E[n][metric]:13.3g}" if n in E else f"{'NA':>13s}" for n in OPTIMIZERS))
    print(f"{'final loss':32s}" + "".join(f"{R[n][3][-1]:13.4f}" if n in E else f"{'NA':>13s}" for n in OPTIMIZERS) + f"   (reference {rl[-1]:.4f})\n")
    if not torch_names: NA.extend((f"model {n} vs PyTorch bf16 {OPTIM}", "no PyTorch bf16 method ran") for n in forge_names); return
    for name in forge_names:
        case(f"model {name} vs PyTorch bf16 {OPTIM}", [(m, E[name][m], BASE * max(E[t][m] for t in torch_names) + 1e-6) for m in E[name]])


def report():
    fails = [r for r in RESULTS if r[2] > r[3]]
    rows = [f"# FORGE correctness ({OPTIM}) : {len(RESULTS) - len(fails)} / {len(RESULTS)} checks pass" + (f", {len(NA)} NA" if NA else ""), "", "| case | check | error | tolerance | result |", "|---|---|--:|--:|:--:|"]
    rows += [f"| {c} | {k} | {e:.3g} | {t:.3g} | {'PASS' if e <= t else '**FAIL**'} |" for c, k, e, t in RESULTS]
    rows += [f"| {c} | {why} | NA | NA | NA |" for c, why in NA]        # could not run : not a pass, not a fail
    os.makedirs(OUT, exist_ok = True); open(f"{OUT}/tests_report.md", "w").write("\n".join(rows) + "\n")
    print(rows[0][2:])
    for c, k, e, t in fails: print(f"  FAIL  {c} : {k} = {e:.3g} > {t:.3g}")


def main():
    module_checks()
    modes = ("bf16", "int8") if OPTIM == "adamw" else (None,)          # sgd : no state, no state_mode
    for mode in modes:
        for wdtype in (torch.float32, torch.bfloat16):
            for shape in SHAPES: layer_step(*shape, wdtype, mode)
        layer_step(*SHAPES[2], torch.bfloat16, mode, scale = 1e-6)    # gradients as small as in a real model
    layer_step(*SHAPES[1], torch.bfloat16, modes[0], bias = True)
    for mode in modes: chain(mode)
    model_check()
    report()


if __name__ == "__main__":
    main()
