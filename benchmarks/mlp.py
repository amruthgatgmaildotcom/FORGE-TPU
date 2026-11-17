import gc, json, math, os, statistics, sys, time, torch, torch.nn as nn, torch.nn.functional as F
from forge_tpu import FusedLinear
from methods import OPTIM, OPTIMIZERS, LABELS                        # the optimizer of this run (OPTIM) and its methods

# Deep residual MLP that learns a fixed random teacher. All parameters are bias-free Linear weights, so FORGE needs no second optimizer.
# Memory at the defaults (1.09B params): AdamW = 2 (w) + 2 (grad) + 4 (m, v) = 8 B/param = 8.7 GB, + ~2.7 GB activations + 0.5 GB data.
# This file also holds the measure + report engine that llama.py and qwen.py use (see "measure" and "report" below).
DEVICE, DTYPE = "tpu", torch.bfloat16
D, FF, LAYERS, BT, BATCHES = 4096, 16384, 8, 4096, 8                 # width, hidden, blocks, tokens per step, batches in the dataset
STEPS, WD = 300, 0.01
LR = {"adamw": 3e-4, "sgd": 1e-2}[OPTIM]
WARMUP, LOG, MAX_SECONDS = 10, 20, float(os.environ.get("MAX_SECONDS", 180))   # a run stops at STEPS or after MAX_SECONDS; the first WARMUP steps are not timed
OUT, GiB = os.environ.get("RESULTS", "results"), 2**30


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Sequential(nn.Linear(D, FF, bias = False), nn.GELU("tanh"), nn.Linear(FF, D, bias = False)) for _ in range(LAYERS))
        self.head = nn.Linear(D, D, bias = False)

    def forward(self, x):
        for block in self.blocks: x = x + block(F.rms_norm(x, (D,)))
        return self.head(F.rms_norm(x, (D,)))


def make_data(seed = 0):
    # BATCHES fixed (x, y) pairs on DEVICE, y = tanh(x A) B : smooth, learnable, the same for every run
    g = torch.Generator().manual_seed(seed)
    A, B = (torch.randn(D, D, generator = g).div(math.sqrt(D)).to(DEVICE, DTYPE) for _ in range(2))
    xs = [torch.randn(BT, D, generator = g).to(DEVICE, DTYPE) for _ in range(BATCHES)]
    return [(x, torch.tanh(x @ A) @ B) for x in xs]


def make_model(name, seed = 0):
    torch.manual_seed(seed)                                            # same init for all four optimizers
    model = MLP().to(DEVICE, DTYPE)
    if name.startswith("forge"):                                       # swap every nn.Linear for a FusedLinear that shares its weight
        fuse(model, name, LR, WD)
        return model, None
    return model, torch_optimizer(name, model.parameters(), LR, WD)


def fuse(module, name, lr, wd):
    # every nn.Linear inside module -> a FusedLinear that shares its weight. forge_bf16 / forge_int8 : AdamW with that state_mode, forge_sgd : SGD (no state, no state_mode)
    kwargs = dict(optimizer_type = "sgd") if name == "forge_sgd" else dict(optimizer_type = "adamw", state_mode = name.split("_")[1])
    for parent in list(module.modules()):
        for child_name, child in parent.named_children():
            if isinstance(child, nn.Linear): setattr(parent, child_name, FusedLinear.from_linear(child, lr = lr, weight_decay = wd, **kwargs))


def torch_optimizer(name, params, lr, wd):
    # the torch optimizer of a method : vanilla (torch's defaults, as the FORGE GPU code), or fused. A FORGE method gets the fused one of its optimizer for the params it does not fuse
    # fused = False is NOT passed : torch then also turns its default foreach off (adam.py, sgd.py : foreach is resolved only when fused and foreach are both None)
    fused = {"fused": True} if name.endswith("_fused") or name.startswith("forge") else {}
    return {"adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}[OPTIM](params, lr = lr, weight_decay = wd, **fused)


def loss_fn(out, y): return F.mse_loss(out.float(), y.float())


#------------------------------------------------------------------------- measure -------------------------------------------------------------------------#

def nbytes(tensors): return sum(t.numel() * t.element_size() for t in tensors)

def state_bytes(model, opt):                                           # torch AdamW states + FORGE states
    torch_states = nbytes(t for s in opt.state.values() for t in s.values() if torch.is_tensor(t)) if opt else 0
    return torch_states + nbytes(t for m in model.modules() if isinstance(m, FusedLinear) for arrays in m.fusedStateObject.buffers.values() for t in arrays.values())


def layer_trace(model, opt, batch, loss_fn, base):
    # one more step (not timed) : memory after every leaf module (Linear, Embedding, norm, activation, ...) in the forward, before each of their backwards, and at loss, end of backward, step.
    # Each point is (memory held at the point, peak since the point before) : the peak also shows short-lived tensors (inside an op, the loss, the step)
    acc, points, hooks = torch.accelerator, [], []
    def mark(label = ""):
        acc.synchronize(); points.append(((acc.memory_allocated() - base) / GiB, (acc.max_memory_allocated() - base) / GiB, label)); acc.reset_peak_memory_stats()
    def after_forward(module, inputs, out):
        mark()
        if torch.is_tensor(out) and out.requires_grad: hooks.append(out.register_hook(lambda grad: mark()))   # a tuple (e.g. RoPE's cos, sin) or a tensor with no grad : no backward point
    hooks += [m.register_forward_hook(after_forward) for m in model.modules() if not any(m.children())]   # every leaf module : the more points, the more precise the line
    x, y = batch
    mark(); points.clear()                                             # starts the peak count
    mark("start"); loss = loss_fn(model(x), y); mark("loss"); loss.backward(); mark("backward")
    if opt: opt.step(); opt.zero_grad(set_to_none = True)
    mark("step")
    for h in hooks: h.remove()
    return points


def measure(name, make_model, data, loss_fn, flops, steps = STEPS):
    # one step = forward + backward + update, timed between two device syncs. Memory : the peak of each step and what is held after it, above the data
    acc = torch.accelerator
    acc.synchronize(); base = acc.memory_allocated()
    model, opt = make_model(name)
    model.train()
    losses, times, peaks, held, start = [], [], [], [], time.perf_counter()
    for step in range(steps):
        x, y = data[step % len(data)]
        acc.reset_peak_memory_stats(); t0 = time.perf_counter()
        loss = loss_fn(model(x), y)
        loss.backward()                                                # FORGE : the fused weights are updated here, inside backward
        if opt: opt.step(); opt.zero_grad(set_to_none = True)
        acc.synchronize(); t1 = time.perf_counter()
        losses.append(loss.item()); times.append((t1 - t0) * 1e3)
        peaks.append((acc.max_memory_allocated() - base) / GiB); held.append((acc.memory_allocated() - base) / GiB)
        if step % LOG == 0: print(f"{name:12s} step {step:4d}  loss {losses[-1]:.4f}  {times[-1]:8.1f} ms  {peaks[-1]:6.2f} GiB", flush = True)
        if t1 - start > MAX_SECONDS: break
    trace = layer_trace(model, opt, data[0], loss_fn, base)
    params, weights, states = sum(p.numel() for p in model.parameters()), nbytes(model.parameters()), state_bytes(model, opt)
    grads = nbytes(p for group in opt.param_groups for p in group["params"]) if opt else 0   # only the params of a torch optimizer get a .grad
    w = WARMUP if len(times) > WARMUP + 1 else 0                       # a very short run : use every step
    peak, tflops = max(peaks[w:]), [flops(params) / t / 1e9 for t in times[w:]]   # t in ms
    del model, opt; gc.collect()
    return dict(name = name, params = params, steps = len(losses), losses = losses, times_ms = times, mem_gib = peaks, held_gib = held, trace = trace, peak = peak,
                weights = weights / GiB, states = states / GiB, grads = grads / GiB, activations = peak - (weights + states + grads) / GiB,
                step_ms = statistics.median(times[w:]), step_std = statistics.pstdev(times[w:]), tflops = statistics.median(tflops), tflops_std = statistics.pstdev(tflops))


#------------------------------------------------------------------------- report --------------------------------------------------------------------------#

REF = OPTIMIZERS[0]                                                    # vanilla torch : the reference of the loss table and of the headline arrows
COLORS = {"adamw": "#2f7ed8", "adamw_fused": "#8c8c8c", "forge_bf16": "#14896b", "forge_int8": "#14896b",           # headline figure
          "sgd": "#2f7ed8", "sgd_fused": "#8c8c8c", "forge_sgd": "#14896b"}
LINES = {"adamw": ("#2f7ed8", "--"), "adamw_fused": ("#8c8c8c", ":"), "forge_bf16": ("#14896b", "-"), "forge_int8": ("#e5484d", "-"),   # line plots
         "sgd": ("#2f7ed8", "--"), "sgd_fused": ("#8c8c8c", ":"), "forge_sgd": ("#14896b", "-")}
PARTS = {"weights": ("weights", "#2f7ed8"), "states": ("opt states", "#1db07a"), "activations": ("activations", "#c9ced8"), "grads": ("weight gradient", "#e5484d")}
RED = "#c62828"


def na_note(fig, missing):
    # a method whose run failed (no result file) is not drawn : say so under the figure
    if missing: fig.text(0.01, 0.005, "NA (the run failed, not drawn) : " + ", ".join(LABELS[n] for n in missing), color = RED, fontsize = 11, ha = "left", va = "bottom")


def at_point(r, label, grads):
    # memory held at one point of the traced step (layer_trace : "loss" = right before backward() starts, "backward" = right when it ends), in parts.
    # weights and states are held all the time, grads = weight gradients live at that point, activations = the rest (activations, logits, ...)
    held = next(p[0] for p in r["trace"] if p[2] == label)
    parts = {"weights": r["weights"], "states": r["states"], "grads": grads}
    return held, {**parts, "activations": max(held - sum(parts.values()), 0.0)}


def headline(R, path, missing = ()):
    import matplotlib.pyplot as plt
    names = list(R)
    fig, (a1, a2, b) = plt.subplots(1, 3, figsize = (21, 5.6), gridspec_kw = {"width_ratios": [1, 1, 1]})
    # (a) and (b) : memory held right before backward() starts (no weight gradient exists yet) and when it ends (torch : all weight gradients), stacked
    points = [(a1, "loss", "(a) Right before backward()", lambda n: 0.0), (a2, "backward", "(b) When backward() ends", lambda n: R[n]["grads"])]
    totals = {(label, n): at_point(R[n], label, grads(n)) for _, label, _, grads in points for n in names}
    top = max(t for t, _ in totals.values()) * 1.35
    forge = [i for i, n in enumerate(names) if n.startswith("forge")]
    for a, label, title, _ in points:
        if forge: a.axvspan(min(forge) - 0.5, max(forge) + 0.5, color = "#e8f4ea", zorder = 0)
        bottom = [0.0] * len(names)
        for key, (part, color) in PARTS.items():
            h = [totals[(label, n)][1][key] for n in names]
            a.bar(range(len(names)), h, 0.62, bottom = bottom, color = color, label = part, zorder = 2)
            bottom = [s + v for s, v in zip(bottom, h)]
        for i, n in enumerate(names): a.text(i, totals[(label, n)][0] * 1.01, f"{totals[(label, n)][0]:.1f}", ha = "center", va = "bottom", fontsize = 14, fontweight = "bold")
        a.set_xticks(range(len(names)), [LABELS[n].replace(" ", "\n") for n in names], fontsize = 13)
        for tick, n in zip(a.get_xticklabels(), names): tick.set_fontweight("bold" if n.startswith("forge") else "normal")
        a.set_ylim(0, top); a.set_xlim(-0.5, len(names) - 0.5)
        a.set_ylabel("Memory held (GiB)", fontsize = 13); a.set_title(title, fontsize = 17, fontweight = "bold")
        a.grid(axis = "y", ls = ":", zorder = 1); a.legend(fontsize = 12, loc = "upper right")
    # (c) memory vs speed (the peak of the timed steps)
    for n in names:
        big = n.startswith("forge")
        b.scatter(R[n]["peak"], R[n]["step_ms"], s = 260 if big else 70, color = COLORS[n], zorder = 3)
        b.annotate(LABELS[n], (R[n]["peak"], R[n]["step_ms"]), xytext = (14, 0) if big else (-10, 12), textcoords = "offset points", va = "center",
                   ha = "left" if big else "right", fontsize = 13 if big else 12, fontweight = "bold" if big else "normal", color = "black" if big else COLORS[n])
    if REF in R:
        for n in (n for n in names if n.startswith("forge")):
            p, q = (R[REF]["peak"], R[REF]["step_ms"]), (R[n]["peak"], R[n]["step_ms"])
            b.annotate("", q, p, arrowprops = dict(arrowstyle = "-|>", color = RED, ls = "--", lw = 2, shrinkA = 8, shrinkB = 12))
            r = p[1] / q[1]
            b.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, f"{r:.2f}× faster" if r >= 1 else f"{1 / r:.2f}× slower", color = RED, fontsize = 13, fontweight = "bold", ha = "center", va = "bottom")
    xs, ys = [R[n]["peak"] for n in names], [R[n]["step_ms"] for n in names]
    pad_x, pad_y = (max(xs) - min(xs)) * 0.35 + 0.5, (max(ys) - min(ys)) * 0.25 + 1
    b.set_xlim(max(xs) + pad_x, min(xs) - pad_x); b.set_ylim(max(ys) + pad_y, min(ys) - pad_y)  # both inverted : better is up and right
    b.set_xlabel("Peak memory (GiB)  ←  smaller", fontsize = 13); b.set_ylabel("Step time (ms)  ↑  faster", fontsize = 13)
    b.set_title("(c) Memory vs. speed", fontsize = 17, fontweight = "bold"); b.grid(ls = ":")
    for ax in (a1, a2, b): ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize = 12)
    fig.tight_layout(rect = (0, 0.04 if missing else 0, 1, 1)); na_note(fig, missing); fig.savefig(path, dpi = 200); plt.close(fig)


def curves(R, key, ylabel, title, path, missing = ()):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize = (10, 5.5))
    for n, r in R.items(): ax.plot(range(r["steps"]), r[key], color = LINES[n][0], ls = LINES[n][1], lw = 2, label = LABELS[n])
    ax.set_xlabel("Training step", fontsize = 13); ax.set_ylabel(ylabel, fontsize = 13); ax.set_title(title, fontsize = 16, fontweight = "bold")
    ax.grid(ls = ":"); ax.legend(fontsize = 12); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect = (0, 0.05 if missing else 0, 1, 1)); na_note(fig, missing); fig.savefig(path, dpi = 200); plt.close(fig)


def memory(R, title, path, missing = ()):
    # (c) the peak of every step (solid) and what it holds after it (dotted)   |   (d) one step, layer by layer, with the peak of the timed steps (thin lines)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, (c, d) = plt.subplots(1, 2, figsize = (16, 5.4))
    for n, r in R.items():
        color, ls = LINES[n]
        c.plot(range(1, r["steps"] + 1), r["mem_gib"], color = color, ls = ls, lw = 2)
        c.plot(range(1, r["steps"] + 1), r["held_gib"], color = color, ls = ":", lw = 1, alpha = 0.7)
        y, top = [p[0] for p in r["trace"]], [p[1] for p in r["trace"]]
        x = list(range(len(y))); x[-1] += max(1, len(y) // 25)        # room for the step
        d.plot(x, y, color = color, ls = ls, lw = 1.5, label = LABELS[n], drawstyle = "steps-post"); d.axhline(r["peak"], color = color, lw = 0.8, alpha = 0.6)   # no slope between two points : held memory stays flat until the next point, then jumps
        d.fill_between(x, y[:1] + y[:-1], top, step = "pre", color = color, alpha = 0.12, lw = 0)   # between two points : from what was held at the first one up to the peak in between
    c.axvspan(0.5, WARMUP + 0.5, color = "#ecebe6", zorder = 0)
    c.text(WARMUP / 2 + 0.5, 0.03, "warmup (not timed)", transform = c.get_xaxis_transform(), ha = "center", color = "#666", fontsize = 11)
    styles = [Line2D([], [], color = LINES[n][0], ls = LINES[n][1], lw = 2, label = LABELS[n]) for n in R]
    styles += [Line2D([], [], color = "#555", lw = 2, label = "peak of the step"), Line2D([], [], color = "#555", ls = ":", lw = 1, label = "held after it")]
    c.legend(handles = styles, ncol = 3, fontsize = 10, loc = "center right")
    trace = next(iter(R.values()))["trace"]
    at = {p[2]: i for i, p in enumerate(trace) if p[2]}
    at["step"] = len(trace) - 1 + max(1, len(trace) // 25)
    for (a, b), name, color in zip(((at["start"], at["loss"]), (at["loss"], at["backward"]), (at["backward"], at["step"])), ("forward", "backward", "step"), ("#eef3f8", "#f8eeee", "#edf5ef")):
        d.axvspan(a, b, color = color, zorder = 0); d.text((a + b) / 2, 0.03, name, transform = d.get_xaxis_transform(), ha = "center", color = "#666", fontsize = 11)
    d.set_xticks([at[k] for k in ("start", "loss", "backward", "step")], ["start", "loss", "backward", "step"])
    for tick, ha in zip(d.get_xticklabels(), ("center", "center", "right", "left")): tick.set_ha(ha)
    d.set_xlabel("line : held at each point   shade : peak between two points   thin lines : peak of the timed steps", color = "#666", fontsize = 10)   # under the plot : at the top it ran into the title
    c.set_xlabel("Step", fontsize = 13); d.legend(fontsize = 11, loc = "upper left")
    c.set_title("(c) Memory at every step", fontsize = 16, fontweight = "bold", loc = "left"); d.set_title("(d) Inside one step, layer by layer", fontsize = 16, fontweight = "bold", loc = "left")
    for ax in (c, d): ax.set_ylabel("Memory (GiB)", fontsize = 13); ax.grid(ls = ":"); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, x = 0.01, ha = "left", color = "#666", fontsize = 11)
    fig.tight_layout(rect = (0, 0.04 if missing else 0, 1, 1)); na_note(fig, missing); fig.savefig(path, dpi = 200); plt.close(fig)


def table(R, desc, missing = ()):
    p = next(iter(R.values()))
    rows = [f"{desc}. Step time is the median of {p['steps'] - WARMUP} steps (± std), after {WARMUP} warmup steps.", "",
            "| Method | Peak (GiB) | Step (ms) | TF/s |", "|---|:--:|:--:|:--:|"]
    order = [n for n in OPTIMIZERS if n in R or n in missing]           # every method in its place, NA where the run failed (e.g. fused SGD not supported on this TPU)
    rows += [f"| {LABELS[n]} | NA | NA | NA |" if n in missing else f"| {LABELS[n]} | {R[n]['peak']:.2f} | {R[n]['step_ms']:.1f} ± {R[n]['step_std']:.1f} | {R[n]['tflops']:.0f} ± {R[n]['tflops_std']:.0f} |" for n in order]
    if REF in R:
        # each run stops at STEPS or MAX_SECONDS, so the runs have different lengths : compare all of them at the last step that every run has
        ref, last = R[REF]["losses"], min(r["steps"] for r in R.values())
        rows += ["", f"| Method | Loss at step {last} (last common step) | max \\|loss − {LABELS[REF]}\\| (steps 1 to {last}) |", "|---|:--:|:--:|"]
        rows += [f"| {LABELS[n]} | NA | NA |" if n in missing else f"| {LABELS[n]} | {R[n]['losses'][last - 1]:.4f} | {max(abs(a - b) for a, b in zip(R[n]['losses'][:last], ref[:last])):.4f} |" for n in order]
    return "\n".join(rows)


def report(prefix, desc, loss_label):
    R = {n: json.load(open(f"{OUT}/{prefix}_{n}.json")) for n in OPTIMIZERS if os.path.exists(f"{OUT}/{prefix}_{n}.json")}
    missing = [n for n in OPTIMIZERS if n not in R]                    # runs that failed : NA in the tables, a note on the plots
    if not R: raise SystemExit(f"{prefix} ({OPTIM}) : no run finished, nothing to report")
    desc = desc.format(params = f"{next(iter(R.values()))['params'] / 1e9:.2f}") + f", {OPTIM}, bf16 weights, single TPU"
    headline(R, f"{OUT}/{prefix}_headline.png", missing)
    curves(R, "losses", loss_label, f"{prefix} ({OPTIM}) : loss vs. training step", f"{OUT}/{prefix}_loss.png", missing)
    memory(R, desc, f"{OUT}/{prefix}_memory.png", missing)
    md = table(R, desc, missing); open(f"{OUT}/{prefix}_table.md", "w").write(md + "\n"); print(md)


def cli(prefix, make_model, make_data, loss_fn, flops, desc, loss_label):
    # python <model>.py <optimizer> : train and measure one optimizer (one process each, so memory is clean)   |   python <model>.py report : plots + table
    os.makedirs(OUT, exist_ok = True)
    if sys.argv[1] == "report": report(prefix, desc, loss_label)
    else: json.dump(measure(sys.argv[1], make_model, make_data(), loss_fn, flops), open(f"{OUT}/{prefix}_{sys.argv[1]}.json", "w"))


if __name__ == "__main__":
    cli("mlp", make_model, make_data, loss_fn, lambda params: 6 * params * BT,                     # 6 N FLOPs per token (forward 2, backward 4)
        f"MLP {{params}}B params ({LAYERS} blocks, D={D}, FF={FF}), {BT} tokens per step", "Loss (MSE)")
