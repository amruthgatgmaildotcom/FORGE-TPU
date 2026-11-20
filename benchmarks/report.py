import glob, json, os, re, shutil, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import METHODS, BEST, LABELS, MODELS, ROOT, RESULTS, ASSETS, run_path, profile_path, tests_path

#   Readme.md
#     python benchmarks/report.py   ->   results/summary.md (every table) and assets/ (every figure). Plain Python + matplotlib : no torch, no TPU, run it anywhere.
#     It reads only what train.py, profiling.py and tests.py wrote (config.py says where), so every number it prints and draws is a recorded one. Several seeds of one
#     method : the numbers are the mean over the seeds, "±" is half the gap between the seeds, the losses are the per-step mean (the band in the plots : min to max).
#        assets/forge_tiling.png                 the kernel, drawn (no data)
#        assets/<model>_headline.png             AdamW : peak memory, split in weights / optimizer states / activations / weight gradient, and memory vs. speed
#        assets/llama8b_headline.png             the same for Llama-3.1-8B on 8 chips, memory of one device (calculated)
#        assets/glance.png                       every model at once against vanilla AdamW : memory saved, speed-up
#        assets/convergence.gif                  Llama-3.2-1B, the losses step by step
#     README.md : benchmarks/README.template.md with every {{key}} (or {{key:format}}) replaced by a number of this file (values()), and every claim of the text
#     checked against the numbers first (claims()) : if one is false, README.md is NOT written.
PEAK_TFLOPS = 197.0                                                                          # TPU v5e, bf16, one chip
REF = {o: m[0] for o, m in METHODS.items()}                                                  # vanilla torch : the reference
ORDER = METHODS
PARTS = [("weights", "weights", "#2b7bd6"), ("states", "opt states", "#1db07a"), ("activations", "activations", "#c8ccd6"), ("grads", "weight gradient", "#e5484d")]
DOT = {"adamw": "#2b7bd6", "sgd": "#2b7bd6", "muon": "#2b7bd6", "adamw_fused": "#8c8c8c", "sgd_fused": "#8c8c8c"}
GREEN, RED = "#12876a", "#d0342c"
LINE = {"adamw": ("black", "--"), "sgd": ("black", "--"), "muon": ("black", "--"), "adamw_fused": ("#8c8c8c", ":"), "sgd_fused": ("#8c8c8c", ":"),
        "forge_bf16": ("#1f77b4", "-"), "forge_sgd": ("#1f77b4", "-"), "forge_muon": ("#1f77b4", "-"), "forge_int8": ("#e8710a", "-")}



#----------------------------------------------------------------------------------------------- data -------------------------------------------------------------------------------------------------#



def runs(optim, model):
    # {method : [run of each seed]} for the methods of optim that have a result
    out = {}
    for m in METHODS[optim]:
        files = sorted(glob.glob(run_path(optim, model, m, "*")), key = lambda f: int(re.search(r"_seed(\d+)\.json$", f).group(1)))
        if files: out[m] = [json.load(open(f)) for f in files]
    return out

def combine(seeds):
    # the runs of one method (one per seed) -> one flat record : means over the seeds, the per-step mean / min / max loss
    n = min(len(r["loss"]) for r in seeds)
    mean = lambda f: statistics.mean(f(r) for r in seeds)
    steps = [r["step_ms"]["median"] for r in seeds]
    return dict(params = seeds[0]["params"], seeds = len(seeds), steps = n, warmup = seeds[0]["settings"]["warmup"], settings = seeds[0]["settings"],
                **{k: mean(lambda r: r["memory_gib"][k]) for k in ("peak", "weights", "states", "grads", "activations")},
                step_ms = statistics.mean(steps), spread = (max(steps) - min(steps)) / 2, step_std = mean(lambda r: r["step_ms"]["std"]), each_ms = steps,
                tflops = mean(lambda r: r["tflops"]["median"]),
                losses = [statistics.mean(r["loss"][i] for r in seeds) for i in range(n)],
                lo = [min(r["loss"][i] for r in seeds) for i in range(n)], hi = [max(r["loss"][i] for r in seeds) for i in range(n)])

def load(optim, model): return {m: combine(s) for m, s in runs(optim, model).items()}

def kaggle_path(optim, method, one_device = True):
    # a Llama-3.1-8B run of benchmarks/kaggle/ : results/kaggle_v5e8/<optim>/llama8b/<method>_seed0.json, and its one-device memory (_1dev)
    return os.path.join(RESULTS, "kaggle_v5e8", os.path.relpath(run_path(optim, "llama8b", method, 0), RESULTS))[:-5] + ("_1dev.json" if one_device else ".json")

def big(optim):
    # Llama-3.1-8B, tensor parallel over 8 TPU v5e chips (one run, seed 0) : step time and loss as measured, the memory of ONE device as CALCULATED
    # by benchmarks/kaggle/one_device.py from the measured memory of one chip (weights, optimizer states, weight gradient x 8, activations x K). "chip_peak" : measured, the largest chip
    out = {}
    for m in METHODS[optim]:
        if not os.path.exists(kaggle_path(optim, m)): continue
        r, chip = json.load(open(kaggle_path(optim, m))), json.load(open(kaggle_path(optim, m, False)))
        out[m] = {**combine([r]), "chip_peak": chip["memory_gib"]["peak"], "chips": chip["chips"]}
    return out

def held(optim, model, method, label):
    # memory held at one labelled point of the traced step ("loss" = right before backward, "backward" = right when it ends), mean over the seeds
    return statistics.mean(next(p[0] for p in r["trace"] if p[2] == label) for r in runs(optim, model)[method])

def peak_phase(optim, model, method):
    # where the traced step of seed 0 reaches its peak : "forward", "start of backward" / "end of backward" (the first / last 5 % of the backward points), "backward", "step"
    t = runs(optim, model)[method][0]["trace"]; at = {p[2]: i for i, p in enumerate(t) if p[2]}
    i, a, b = max(range(len(t)), key = lambda k: t[k][1]), at["loss"], at["backward"]
    if i <= a: return "forward"
    if i > b: return "step"
    return "start of backward" if i - a <= 0.05 * (b - a) else "end of backward" if b - i <= 0.05 * (b - a) else "backward"

def profiles(optim, model):
    return {m: json.load(open(profile_path(optim, model, m))) for m in METHODS[optim] if os.path.exists(profile_path(optim, model, m))}




#---------------------------------------------------------------------------------------------- figures -----------------------------------------------------------------------------------------------#



def pct(a, b): return (1 - a / b) * 100                                                                                              # % less of a than b

def x_(a, b): return b / a                                                                                                           # how many times faster a is than b (step times)

def smooth(y, span):
    h = span // 2
    return [sum(y[max(0, k - h): k + h + 1]) / len(y[max(0, k - h): k + h + 1]) for k in range(len(y))]

def span_of(n): return max(3, n // 30)

def tail(y): return statistics.mean(y[-max(1, len(y) // 10):])                                                                      # the final loss : the mean of the last 10 % of the steps

def plt_():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    return plt



#--------------------------------------------------------------------------------------------- figures -----------------------------------------------------------------------------------------------#

def headline(R, optim, path, title_a, title_b = "(b) Memory vs. speed", unit = "GiB"):
    # (a) the peak of a step, split in weights, optimizer states, activations, weight gradient (stacked)   |   (b) peak memory vs. step time, better is up and right
    plt = plt_()
    names = list(R)
    fig, (a, b) = plt.subplots(1, 2, figsize = (12.6, 5.0), gridspec_kw = {"width_ratios": [1.05, 1]})
    forge = [i for i, n in enumerate(names) if n.startswith("forge")]
    if forge: a.axvspan(min(forge) - 0.5, max(forge) + 0.5, color = "#e7f2ea", zorder = 0)
    bottom = [0.0] * len(names)
    for key, label, color in PARTS:
        h = [max(R[n][key], 0.0) for n in names]
        a.bar(range(len(names)), h, 0.62, bottom = bottom, color = color, label = label, zorder = 2)
        bottom = [s + v for s, v in zip(bottom, h)]
    top = max(bottom)
    for i, n in enumerate(names): a.text(i, bottom[i] + top * 0.015, f"{R[n]['peak']:.1f}", ha = "center", va = "bottom", fontsize = 15, fontweight = "bold")
    a.set_xticks(range(len(names)), [LABELS[n].replace(" (", "\n(").replace("vanilla ", "vanilla\n").replace("fused ", "fused\n") for n in names], fontsize = 13)
    for tick, n in zip(a.get_xticklabels(), names): tick.set_fontweight("bold" if n.startswith("forge") else "normal")
    a.set_ylim(0, top * 1.34); a.set_xlim(-0.5, len(names) - 0.5)
    a.set_ylabel(f"Peak memory ({unit})", fontsize = 14); a.set_title(title_a, fontsize = 17, fontweight = "bold", loc = "left")
    a.grid(axis = "y", ls = ":", color = "#bbbbbb", zorder = 1); a.legend(fontsize = 11.5, loc = "upper right", framealpha = 0.95); a.tick_params(labelsize = 12)
    ref = REF[optim]
    for n in names:
        big_ = n.startswith("forge")
        b.scatter(R[n]["peak"], R[n]["step_ms"], s = 300 if big_ else 90, color = GREEN if big_ else DOT.get(n, "#8c8c8c"), zorder = 4)
    xs, ys = [R[n]["peak"] for n in names], [R[n]["step_ms"] for n in names]
    px, py = (max(xs) - min(xs)) * 0.30 + 0.05 * max(xs), (max(ys) - min(ys)) * 0.9 + 0.03 * max(ys)
    b.set_xlim(max(xs) + px, min(xs) - px); b.set_ylim(max(ys) + py, min(ys) - py)                                                      # both inverted : smaller memory to the right, faster up
    for n in names:
        big_ = n.startswith("forge")
        b.annotate(LABELS[n].replace("FORGE (bf16)", "FORGE").replace("FORGE (SGD)", "FORGE").replace("FORGE (Muon)", "FORGE"), (R[n]["peak"], R[n]["step_ms"]),
                   xytext = (16, 14) if big_ else ((12, -14) if n == ref else (-10, 12)), textcoords = "offset points", ha = "left" if big_ or n == ref else "right", va = "center",
                   fontsize = 15 if big_ else 13, fontweight = "bold" if big_ else "normal", color = "black" if big_ else DOT.get(n, "#777777"))
    if ref in R:
        p, gains = (R[ref]["peak"], R[ref]["step_ms"]), []
        for n in (n for n in names if n.startswith("forge")):
            q = (R[n]["peak"], R[n]["step_ms"])
            b.annotate("", q, p, arrowprops = dict(arrowstyle = "-|>", color = RED, ls = (0, (4, 3)), lw = 2.2, shrinkA = 9, shrinkB = 14), zorder = 3)
            gains.append(f"{LABELS[n]} :  {x_(q[1], p[1]):.2f}×  faster,  {pct(q[0], p[0]):.0f}% less memory")
        b.text(0.97, 0.05, f"against {LABELS[ref]}\n" + "\n".join(gains), transform = b.transAxes, ha = "right", va = "bottom", color = RED, fontsize = 12, fontweight = "bold", linespacing = 1.5,
               bbox = dict(boxstyle = "round,pad=0.5", fc = "white", ec = "#e3b5b2"))
    b.set_xlabel(f"Peak memory ({unit})   ←  smaller", fontsize = 14); b.set_ylabel("Step time (ms)   ↑  faster", fontsize = 14)
    b.set_title(title_b, fontsize = 17, fontweight = "bold", loc = "left"); b.grid(ls = ":", color = "#cccccc"); b.tick_params(labelsize = 12)
    fig.tight_layout(); fig.savefig(path, dpi = 170); plt.close(fig)

def glance(rows, path):
    # every model at once, against vanilla AdamW : (a) peak memory saved by FORGE (bf16, int8) and by fused AdamW   |   (b) step speed-up
    plt = plt_()
    import numpy as np
    names = [r[0] for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, (a, b) = plt.subplots(1, 2, figsize = (13, 0.62 * len(rows) + 1.9), sharey = True)
    for ax, key, title, unit in ((a, "mem", "(a) Peak memory saved", "%"), (b, "speed", "(b) Step speed-up", "×")):
        bars = [("fused AdamW", "#b9b9b9", -0.27, "f"), ("FORGE (bf16)", GREEN, 0.0, "g"), ("FORGE (int8)", "#e8710a", 0.27, "i")]
        for label, color, dy, k in bars:
            vals = [r[1][key].get(k) for r in rows]
            ax.barh([yy + dy for yy, v in zip(y, vals) if v is not None], [v for v in vals if v is not None], 0.25, color = color, label = label, zorder = 2)
            for yy, v in zip(y, vals):
                if v is None: continue
                ax.text(v + (0.6 if key == "mem" else 0.012) if v >= (0 if key == "mem" else 1) else (0.6 if key == "mem" else 1.012), yy + dy,
                        f"{v:.0f}%" if key == "mem" else f"{v:.2f}×", va = "center", fontsize = 9.5, fontweight = "bold" if k != "f" else "normal", color = "#333333")
        ax.set_title(title + " against vanilla AdamW", fontsize = 14, fontweight = "bold", loc = "left"); ax.grid(axis = "x", ls = ":", zorder = 1)
        ax.tick_params(labelsize = 11)
    a.set_yticks(y, names, fontsize = 11.5); a.set_xlabel("% less peak memory", fontsize = 12); a.axvline(0, color = "black", lw = 0.8)
    b.set_xlabel("× faster (step time of vanilla AdamW / step time)", fontsize = 12); b.axvline(1, color = "black", lw = 0.8)
    b.set_xlim(left = min(0.9, min(r[1]["speed"].get(k, 1) for r in rows for k in "fgi") - 0.05))
    a.set_xlim(left = min(0, min(r[1]["mem"].get(k, 0) for r in rows for k in "fgi") - 2))
    a.legend(fontsize = 10.5, loc = "lower right")
    fig.tight_layout(); fig.savefig(path, dpi = 160); plt.close(fig)

def gif(R, optim, title, path, frames = 50):
    # the convergence animation : the smoothed losses grow step by step, the raw losses behind them
    plt = plt_()
    from matplotlib.animation import FuncAnimation, PillowWriter
    n = min(r["steps"] for r in R.values()); s = span_of(n)
    order = [m for m in ORDER[optim] if m in R]
    sm = {m: smooth(R[m]["losses"][:n], s) for m in order}
    hi = max(sorted(R[m]["losses"][:n])[int(n * 0.99) - 1] for m in order) * 1.05
    fig, ax = plt.subplots(figsize = (8.0, 4.3))
    raws = {m: ax.plot([], [], color = "#a99aa8", lw = 0.7, alpha = 0.5)[0] for m in order}
    lines = {m: ax.plot([], [], color = LINE[m][0], ls = LINE[m][1], lw = 2.4 if m.startswith("forge") else 2.0, label = LABELS[m])[0] for m in order}
    ax.set_xlim(0, n); ax.set_ylim(0, hi); ax.grid(alpha = 0.35); ax.legend(loc = "upper right", fontsize = 10)
    ax.set_xlabel("training step"); ax.set_ylabel("training loss (nats)"); ax.set_title(title, fontsize = 12, fontweight = "bold")
    def draw(f):
        k = min(n, max(2, int(n * (f + 1) / frames)))
        for m in order: raws[m].set_data(range(k), R[m]["losses"][:k]); lines[m].set_data(range(k), sm[m][:k])
        return [*raws.values(), *lines.values()]
    FuncAnimation(fig, draw, frames = frames + 12, blit = False).save(path, writer = PillowWriter(fps = 12), dpi = 90)
    plt.close(fig)



#---------------------------------------------------------------------------------------------- diagram -----------------------------------------------------------------------------------------------#



def tiling(path):
  # FORGE on TPU : the memory hierarchy of one v5e chip, the (bv, bh, bbt) grid of the Pallas kernel, and the update of one tile in VMEM
    plt = plt_()
    from matplotlib.patches import Polygon, Rectangle, FancyBboxPatch, FancyArrowPatch
    fig = plt.figure(figsize = (16, 6.6)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 160); ax.set_ylim(0, 66); ax.axis("off")
    teal, teal_e, gold, gold_e, pink, pink_e = "#b2e3e6", "#2a8a8f", "#fff1c9", "#d4a73a", "#f7d1d1", "#c9504d"
  # left : the memory hierarchy (a pyramid)
    ax.text(24, 59, "TPU v5e chip\nmemory hierarchy", ha = "center", va = "center", fontsize = 17, fontweight = "bold")
    for (y0, y1, w0, w1, fc, ec, name, sub) in ((42, 51, 9, 4, gold, gold_e, "vregs", "8 × 128 lanes · 1 cycle"),
                                               (27, 38, 17, 11, pink, pink_e, "VMEM", "128 MiB on-chip scratchpad"),
                                               (9, 23, 24, 18, teal, teal_e, "HBM", "16 GiB · 819 GB/s")):
        ax.add_patch(Polygon([(24 - w0, y0), (24 + w0, y0), (24 + w1, y1), (24 - w1, y1)], closed = True, fc = fc, ec = ec, lw = 1.4))
        ax.text(24, (y0 + y1) / 2 + 1.6, name, ha = "center", va = "center", fontsize = 15, fontweight = "bold")
        ax.text(24, (y0 + y1) / 2 - 2.0, sub, ha = "center", va = "center", fontsize = 11.5)
  # middle : X (top), dY^T (left), the dW grid (never materialised), one tile in VMEM
    gx, gy, gw, gh = 70, 10, 34, 34                                                    # the dW (V x H) square
    ax.add_patch(Rectangle((gx, gy), gw, gh, fc = "white", ec = "black", lw = 1.2))
    ax.text(gx + gw - 1, gy + 1.5, r"$\nabla_W\mathcal{L}\in\mathbb{R}^{V\times H}$  (never in HBM)", ha = "right", fontsize = 11, color = "#9a9a9a")
    ax.add_patch(Rectangle((gx, 50), gw, 9, fc = "white", ec = "black", lw = 1.2))     # X (BT x H) : columns are H
    for i in range(1, 5): ax.plot([gx + gw * i / 5] * 2, [50, 59], color = "black", lw = 0.8)
    ax.add_patch(Rectangle((gx + gw * 2 / 5, 50), gw / 5, 9, fc = pink, ec = pink_e, lw = 1.4))
    ax.text(gx + gw / 2, 61.2, r"$X\in\mathbb{R}^{BT\times H}$", ha = "center", fontsize = 14)
    ax.text(gx + gw * 2.5 / 5, 47.7, r"$b_h$", ha = "center", fontsize = 12)
    ax.add_patch(Rectangle((54, gy), 9, gh, fc = "white", ec = "black", lw = 1.2))     # dY^T (V x BT) : rows are V
    for i in range(1, 5): ax.plot([54, 63], [gy + gh * i / 5] * 2, color = "black", lw = 0.8)
    ax.add_patch(Rectangle((54, gy + gh * 2 / 5), 9, gh / 5, fc = pink, ec = pink_e, lw = 1.4))
    ax.text(50.5, gy + gh / 2, r"$\Delta Y^{\top}\in\mathbb{R}^{V\times BT}$", rotation = 90, ha = "center", va = "center", fontsize = 14)
    ax.text(64.6, gy + gh * 2.5 / 5, r"$b_v$", va = "center", fontsize = 12)
    ax.annotate("", (63, gy + gh + 1.2), (54, gy + gh + 1.2), arrowprops = dict(arrowstyle = "->", lw = 1.2)); ax.text(58.5, gy + gh + 2.2, r"$b_{bt}$ (k axis)", ha = "center", fontsize = 11)
    tx, ty = gx + gw * 2 / 5, gy + gh * 2 / 5
    ax.add_patch(Rectangle((tx, ty), gw / 5, gh / 5, fc = gold, ec = gold_e, lw = 2.4))
    ax.text(tx + gw / 10, ty + gh / 10 + 0.9, r"$G_{ij}$", ha = "center", va = "center", fontsize = 14, fontweight = "bold")
    ax.text(tx + gw / 10, ty + gh / 10 - 1.9, r"$b_v\times b_h$", ha = "center", va = "center", fontsize = 9)
    ax.plot([tx, tx], [ty + gh / 5, 50], color = "black", ls = ":", lw = 1); ax.plot([tx + gw / 5] * 2, [ty + gh / 5, 50], color = "black", ls = ":", lw = 1)
    ax.plot([63, tx], [ty + gh / 5] * 2, color = "black", ls = ":", lw = 1); ax.plot([63, tx], [ty, ty], color = "black", ls = ":", lw = 1)
    ax.text(tx + gw / 10, ty - 5.0, "MXU :  " + r"$G_{ij}$ += $\Delta Y_{k}^{\top}X_{k}$" + "  (fp32, in VMEM)", ha = "center", va = "center", fontsize = 11)
    ax.text(gx + gw / 2, 4.2, "Pallas grid  (V / b_v,  H / b_h,  BT / b_bt)  —  FORGE", ha = "center", fontsize = 14, fontweight = "bold")
  # right : the update of the tile, in VMEM
    bx, by, bw, bh_ = 117, 13, 40, 33
    ax.text(bx + bw / 2, 62.5, "In VMEM (one tile)", ha = "center", fontsize = 15, fontweight = "bold", color = teal_e)
    ax.text(bx + bw / 2, 58.0, "read from HBM", ha = "center", fontsize = 11.5)
    for k, (x, lab) in enumerate(((bx + 4, r"$W_{ij}$"), (bx + 17, r"$m_{ij}$"), (bx + 29, r"$v_{ij}$"))):
        ax.add_patch(FancyBboxPatch((x, 50), 8.5 if k else 10, 5.5, boxstyle = "round,pad=0.3,rounding_size=1.0", fc = teal, ec = teal_e, lw = 1.3))
        ax.text(x + (4.25 if k else 5), 52.75, lab, ha = "center", va = "center", fontsize = 14)
        ax.add_patch(FancyBboxPatch((x, 3.5), 8.5 if k else 10, 5.5, boxstyle = "round,pad=0.3,rounding_size=1.0", fc = teal, ec = teal_e, lw = 1.3))
        ax.text(x + (4.25 if k else 5), 6.25, lab, ha = "center", va = "center", fontsize = 14)
        for xx in (x + 2.5, x + (7.5 if not k else 6)):
            ax.annotate("", (xx, by + bh_ + 0.3), (xx, 49.6), arrowprops = dict(arrowstyle = "->", color = teal_e, lw = 1.1))
            ax.annotate("", (xx, 9.4), (xx, by - 0.3), arrowprops = dict(arrowstyle = "->", color = teal_e, lw = 1.1))
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh_, boxstyle = "square,pad=0", fc = gold, ec = gold_e, lw = 2.6))
    ax.text(bx + bw / 2, by + bh_ - 3.5, "Optimizer update", ha = "center", fontsize = 14, fontweight = "bold")
    ax.add_patch(FancyBboxPatch((bx + 3, by + 6), bw - 6, 19, boxstyle = "round,pad=0.2,rounding_size=1.5", fc = "white", ec = "#444444", ls = "--", lw = 1.1))
    ax.text(bx + bw / 2, by + 21.5, r"$m_{ij}\leftarrow\beta_1 m_{ij}+(1-\beta_1)G_{ij}$", ha = "center", fontsize = 12)
    ax.text(bx + bw / 2, by + 16.5, r"$v_{ij}\leftarrow\beta_2 v_{ij}+(1-\beta_2)G_{ij}^{\odot 2}$", ha = "center", fontsize = 12)
    ax.text(bx + bw / 2, by + 11.0, r"$W_{ij}\leftarrow W_{ij}(1-\eta\lambda)-\frac{\eta}{b_1}\,\frac{m_{ij}}{\sqrt{v_{ij}/b_2}+\epsilon}$", ha = "center", fontsize = 12)
    ax.text(bx + bw / 2, by + 2.2, "written back in place (input_output_aliases)", ha = "center", fontsize = 9.5, color = "#444444")
    ax.text(bx + bw / 2, 0.6, "same HBM buffers", ha = "center", fontsize = 11.5)
    ax.add_patch(FancyArrowPatch((tx + gw / 5 + 0.5, ty + gh / 5), (bx - 0.5, by + bh_ - 2), arrowstyle = "-|>", mutation_scale = 14, ls = "--", lw = 1.3, color = "black"))
    ax.add_patch(FancyArrowPatch((tx + gw / 5 + 0.5, ty), (bx - 0.5, by + 2), arrowstyle = "-|>", mutation_scale = 14, ls = "--", lw = 1.3, color = "black"))
    fig.savefig(path, dpi = 140, facecolor = "white"); plt.close(fig)




#---------------------------------------------------------------------------------------------- numbers ----------------------------------------------------------------------------------------------#



def compare(R, optim):
    # every method of R with its comparisons : dmem = % less peak than vanilla torch (x : times faster), dmem_best / x_best : against the best torch method
    if not R: return R
    ref, best = R.get(REF[optim]), R.get(BEST[optim])
    n = min(r["steps"] for r in R.values())                                                  # the final loss of every method over the same steps (runs stop on a time limit)
    for m, r in R.items():
        r.update(mfu = r["tflops"] / PEAK_TFLOPS * 100 / r.get("chips", 1), loss = tail(r["losses"][:n]), common_steps = n, timed = r["steps"] - r["warmup"])
        if ref:  r.update(dmem = pct(r["peak"], ref["peak"]),  x = x_(r["step_ms"], ref["step_ms"]), dstep = pct(r["step_ms"], ref["step_ms"]))
        if best: r.update(dmem_best = pct(r["peak"], best["peak"]), x_best = x_(r["step_ms"], best["step_ms"]), dstep_best = pct(r["step_ms"], best["step_ms"]))
    return R

def values():
    # every number the README can show, as {"key": value}. Keys : <model>.<method>.<field> (AdamW), muon.<model>.<method>.<field>, 8b.<method>.<field>,
    # 8b_muon.<method>.<field>, prof.<model>.<method>.<field>, and a few derived ones (see the end)
    V, D = {}, {"adamw": {}, "muon": {}}
    def put(prefix, R):
        for m, r in R.items():
            for k, v in r.items():
                if isinstance(v, (int, float)): V[f"{prefix}{m}.{k}"] = v
    for model in MODELS:
        for optim in ("adamw", "muon"):
            R = compare(load(optim, model), optim)
            if R: D[optim][model] = R; put(f"{model}." if optim == "adamw" else f"muon.{model}.", R)
        for m, p in profiles("adamw", model).items():
            s, o = p["per_phase"], p["one_sync"]
            V.update({f"prof.{model}.{m}.{k}": v for k, v in dict(fwd = s["forward"]["done"], bwd = s["backward"]["done"], bwd_issued = s["backward"]["issued"], opt = s["step"]["done"],
                                                                     step = o["done"], issued = o["step"], py = o["forge_python"], updates = p["forge_updates"]).items()})
    D["8b"], D["8b_muon"] = compare(big("adamw"), "adamw"), compare(big("muon"), "muon")
    put("8b.", D["8b"]); put("8b_muon.", D["8b_muon"])
    for key in ("8b", "8b_muon"):
        ref = D[key].get(REF["adamw" if key == "8b" else "muon"])
        for m, r in D[key].items(): V[f"{key}.{m}.saved"] = ref["peak"] - r["peak"]
    # derived
    V["seq_8b"] = D["8b"]["forge_bf16"]["settings"]["seq"]
    V["k_1dev"] = json.load(open(kaggle_path("adamw", "forge_bf16")))["one_device"]["K"]               # the activation factor of the one-device calculation
    V["qwen.bwd_end_saved"] = held("adamw", "qwen", "adamw_fused", "backward") - held("adamw", "qwen", "forge_bf16", "backward")
    V["qwen.bwd_extra"] = V["prof.qwen.forge_bf16.bwd"] - V["prof.qwen.adamw_fused.bwd"]
    V["qwen.opt_saved"] = V["prof.qwen.adamw_fused.opt"] - V["prof.qwen.forge_bf16.opt"]
    V["llama.int8_bwd_extra"] = V["prof.llama.forge_int8.bwd"] - V["prof.llama.forge_bf16.bwd"]
    V["llama.peak_below_fused"] = held("adamw", "llama", "adamw_fused", "backward") - held("adamw", "llama", "forge_bf16", "backward")
    V["vocab.qwen"] = 151936
    V["llama.opt_cut"] = pct(V["prof.llama.forge_bf16.opt"], V["prof.llama.adamw.opt"]); V["llama.opt_cut_best"] = pct(V["prof.llama.forge_bf16.opt"], V["prof.llama.adamw_fused.opt"])
    for key, R in (("1b", D["adamw"]["llama"]), ("8b", D["8b"])):                         # how much of the deleted gradient pool leaves the peak : small model against large model
        f, g = R["adamw_fused"], R["forge_bf16"]
        V[f"pool.{key}"], V[f"pool_saved.{key}"] = f["grads"] - g["grads"], f["peak"] - g["peak"]
        V[f"pool_pct.{key}"], V[f"act_per_weight.{key}"] = (f["peak"] - g["peak"]) / (f["grads"] - g["grads"]) * 100, f["activations"] / f["weights"]
    V["qwen.forge_bf16_slower"] = -D["adamw"]["qwen"]["forge_bf16"]["dstep_best"]
    V["llama.int8_slower_best"] = -D["adamw"]["llama"]["forge_int8"]["dstep_best"]
    V["llama.int8_more_saved"] = D["adamw"]["llama"]["forge_bf16"]["peak"] - D["adamw"]["llama"]["forge_int8"]["peak"]
    gaps = {k: (abs(D["adamw"][k]["forge_bf16"]["loss"] - D["adamw"][k]["adamw"]["loss"]), abs(D["adamw"][k]["adamw_fused"]["loss"] - D["adamw"][k]["adamw"]["loss"])) for k in D["adamw"]}
    V["same.gap"] = max(gaps[k][0] for k in ("mlp", "llama"))
    t = open(tests_path("adamw")).read().splitlines()
    rows = [l for l in t if l.startswith("|") and ("PASS" in l or "FAIL" in l)]
    fails = [l for l in rows if "**FAIL**" in l]
    V.update({"tests.total": len(rows), "tests.pass": len(rows) - len(fails), "tests.fails": len(fails), "tests.int8_fails": sum("int8" in l for l in fails)})
    return V, D, gaps, fails



#---------------------------------------------------------------------------------------------- claims -----------------------------------------------------------------------------------------------#



def claims(V, D, gaps, fails):
    # every sentence of the README that is not a plain number, checked against the numbers. [(true ?, the claim)]
    a, m, b8 = D["adamw"], D["muon"], D["8b"]
    return [(a["llama"]["forge_bf16"]["step_ms"] < a["llama"]["adamw_fused"]["step_ms"] and a["llama"]["forge_bf16"]["peak"] < a["llama"]["adamw_fused"]["peak"], "Llama-3.2-1B : FORGE (bf16) is faster AND smaller than fused AdamW"),
            (a["mlp"]["forge_bf16"]["step_ms"] < a["mlp"]["adamw_fused"]["step_ms"] and a["mlp"]["forge_int8"]["step_ms"] < a["mlp"]["adamw_fused"]["step_ms"], "MLP : both FORGE arms are faster than fused AdamW"),
            (peak_phase("adamw", "llama", "adamw_fused") == "end of backward" and peak_phase("adamw", "llama", "forge_bf16") == "start of backward", "Llama-3.2-1B : fused AdamW peaks at the end of backward, FORGE at the start of backward"),
            (D["adamw"]["qwen"]["forge_int8"]["dmem_best"] > D["adamw"]["qwen"]["forge_bf16"]["dmem_best"] > 0, "Qwen3-0.6B : bf16 lowers the peak a little, int8 more"),
            (all(peak_phase(o, "qwen", x) == "start of backward" for o, x in (("adamw", "adamw_fused"), ("adamw", "forge_bf16"), ("muon", "muon"), ("muon", "forge_muon"))), "Qwen3-0.6B : every method peaks at the start of backward"),
            (V["qwen.bwd_extra"] > V["qwen.opt_saved"], "Qwen3-0.6B : FORGE's longer backward costs more than its shorter optimizer step saves"),
            (a["qwen"]["forge_bf16"]["step_ms"] > a["qwen"]["adamw_fused"]["step_ms"], "Qwen3-0.6B : FORGE (bf16) is slower than fused AdamW"),
            (a["llama"]["forge_int8"]["step_ms"] > a["llama"]["forge_bf16"]["step_ms"] and V["llama.int8_bwd_extra"] > 0, "Llama-3.2-1B : int8 is slower than bf16, its backward is longer"),
            (a["llama"]["forge_int8"]["step_ms"] > a["llama"]["adamw_fused"]["step_ms"], "Llama-3.2-1B : int8 is slower than fused AdamW"),
            (b8["forge_bf16"]["peak"] < b8["adamw_fused"]["peak"] and b8["forge_bf16"]["step_ms"] < b8["adamw_fused"]["step_ms"], "Llama-3.1-8B : FORGE (bf16) is faster AND smaller than fused AdamW"),
            (max(b8[x]["step_ms"] for x in ("forge_bf16", "forge_int8")) < min(b8[x]["step_ms"] for x in ("adamw", "adamw_fused")), "Llama-3.1-8B : both FORGE arms are faster than both torch methods"),
            (m["llama"]["forge_muon"]["peak"] < m["llama"]["muon"]["peak"] and m["llama"]["forge_muon"]["step_ms"] < m["llama"]["muon"]["step_ms"], "Llama-3.2-1B : FORGE (Muon) is faster AND smaller than Muon"),
            (all(g <= f + 1e-4 for g, f in gaps.values()), "every model : the FORGE (bf16) final loss is never further from vanilla AdamW than fused AdamW's"),
            (all("int8" in l or "| db |" in l for l in fails), "tests : every fail is an int8 case or the bias gradient db"),
            (all(r["dmem"] > 0 for k in a for x, r in a[k].items() if x.startswith("forge")) and all(r["dmem"] > 0 for k in m for x, r in m[k].items() if x.startswith("forge")), "every model : FORGE has less peak memory than vanilla torch"),
            (all(a[k][x]["dstep"] > 0 for k in a for x in ("forge_bf16", "forge_int8")) and all(a[k]["forge_bf16"]["dstep_best"] > 0 for k in a if k != "qwen"), "one chip : FORGE is faster than vanilla AdamW on each model, and faster than fused AdamW except on Qwen3-0.6B"),
            (D["8b"]["forge_bf16"]["mfu"] < 10, "Llama-3.1-8B : the 8 chips reach less than 10% of their peak"),
            ([D["8b"][x]["settings"]["warmup"] for x in D["8b"]] == [10] * len(D["8b"]) and all(D["8b"][x]["settings"]["lr_warmup"] == 0 and D["8b"][x]["steps"] == 200 for x in D["8b"]), "Llama-3.1-8B : 200 steps, 10 not timed, no lr warmup"),
            (V["pool_pct.8b"] > V["pool_pct.1b"] and V["act_per_weight.8b"] < V["act_per_weight.1b"], "a larger model : more of the deleted gradient pool leaves the peak, activations are smaller next to the weights"),
            (a["mlp"]["forge_bf16"]["dstep_best"] > a["llama"]["forge_bf16"]["dstep_best"], "larger layers : a larger speed-up against fused AdamW (MLP against Llama-3.2-1B)")]



#---------------------------------------------------------------------------------------------- tables -----------------------------------------------------------------------------------------------#



def tables(V, D):
    # the tables of the README, as markdown strings (key "table.<name>")
    B = lambda m, x: f"**{x}**" if m.startswith("forge") else x
    T = {}
    R = D["adamw"]["llama"]
    vs = lambda m, r, k, d = 0: "" if (m == "adamw" or (k == "_best" and m == "adamw_fused")) else B(m, f"{-r['dmem' + k]:+.{d}f}% memory · {-r['dstep' + k]:+.{d}f}% time")
    T["table.llama"] = "\n".join(["| Method | Peak (GiB) | weight gradient (GiB) | Step (ms) | vs. vanilla AdamW | vs. fused AdamW |", "|---|:--:|:--:|:--:|:--:|:--:|"] +
                                 [f"| {B(m, LABELS[m])} | {B(m, f'{r['peak']:.2f}')} | {B(m, f'{r['grads']:.2f}')} | {B(m, f'{r['step_ms']:.1f} ± {r['spread']:.1f}')} | {vs(m, r, '')} | {vs(m, r, '_best')} |" for m, r in R.items()])
    T["table.8b"] = "\n".join(["| Method | Peak of one device, calculated (GiB) | Step on 8 chips (ms) | vs. vanilla AdamW | vs. fused AdamW |", "|---|:--:|:--:|:--:|:--:|"] +
                              [f"| {B(m, LABELS[m])} | {B(m, f'{r['peak']:.1f}')} | {B(m, f'{r['step_ms']:.1f}')} | {vs(m, r, '')} | {vs(m, r, '_best')} |" for m, r in D["8b"].items()])
    rows = ["| Model | Method | Peak (GiB) | Step (ms) | vs. vanilla AdamW | vs. fused AdamW |", "|---|---|:--:|:--:|:--:|:--:|"]
    for model, title in MODELS.items():
        for k, (m, r) in enumerate(D["adamw"][model].items()):
            vs = "" if m == "adamw" else B(m, f"{-r['dmem']:+.0f}% memory · {-r['dstep']:+.0f}% time")
            vb = "" if m in ("adamw", "adamw_fused") else f"{-r['dmem_best']:+.1f}% memory · {-r['dstep_best']:+.1f}% time"
            rows.append(f"| {f'**{title}**' if k == 0 else ''} | {B(m, LABELS[m])} | {B(m, f'{r['peak']:.2f}')} | {B(m, f'{r['step_ms']:.1f}')} | {vs} | {vb} |")
    T["table.models"] = "\n".join(rows)
    rows = ["| Model | Method | Forward | Backward | **Optimizer step** | Step, as trained |", "|---|---|:--:|:--:|:--:|:--:|"]
    for model in ("llama", "qwen"):
        for k, m in enumerate(METHODS["adamw"]):
            p = lambda f: V[f"prof.{model}.{m}.{f}"]
            opt, step = f"{p('opt'):.1f}", f"{p('step'):.1f}"
            rows.append(f"| {MODELS[model] if k == 0 else ''} | {B(m, LABELS[m])} | {p('fwd'):.1f} | {p('bwd'):.1f} | {B(m, opt)} | {B(m, step)} |")
    T["table.profile"] = "\n".join(rows)
    T["table.loss"] = "\n".join(["| Model | Steps | " + " | ".join(LABELS[m] for m in METHODS["adamw"]) + " |", "|---|:--:|" + ":--:|" * 4] +
                                [f"| {MODELS[k]} | {R['adamw']['common_steps']} | " + " | ".join(B(m, f"{R[m]['loss']:.4f}") for m in METHODS["adamw"]) + " |" for k, R in D["adamw"].items()])
    rows = ["| Model | Method | Peak (GiB) | Step (ms) | vs. standard Muon |", "|---|---|:--:|:--:|:--:|"]
    for title, R in [(MODELS[k], D["muon"][k]) for k in D["muon"]] + [(f"Llama-3.1-8B, one device (calculated), 8 chips", D["8b_muon"])]:
        for k, (m, r) in enumerate(R.items()):
            rows.append(f"| {title if k == 0 else ''} | {B(m, LABELS[m])} | {B(m, f'{r['peak']:.2f}')} | {B(m, f'{r['step_ms']:.1f}')} | {'' if m == 'muon' else B(m, f"{-r['dmem']:+.1f}% memory · {-r['dstep']:+.1f}% time")} |")
    T["table.muon"] = "\n".join(rows)
    return T

def summary():
    # results/summary.md : every table, from the recorded runs only
    out = ["# Results", "", "Written by `benchmarks/report.py` from the files in `results/`. One TPU v5e chip (197 TF/s bf16). Mean over the seeds; ± : half the gap between the seeds "
           "(one seed : ± the std of the timed steps). MFU = TF/s / 197.", ""]
    for optim in METHODS:
        for model, title in MODELS.items():
            R, S = load(optim, model), runs(optim, model)
            if not R: continue
            r0 = next(iter(R.values())); st = r0["settings"]
            out += [f"## {title} · {optim}", "", f"{r0['params'] / 1e9:.2f}B params, {st['tokens_per_step']} tokens per step, lr {st['lr']:g}, {r0['seeds']} seed(s), "
                    f"{st['warmup']} warmup steps (not timed), up to {st['max_steps']} steps or {st['max_seconds']:.0f} s.", "",
                    "| Method | Peak (GiB) | Step (ms) | TF/s | MFU | Peak vs. vanilla | Step vs. vanilla | Peak vs. best torch | Step vs. best torch | Final loss |", "|---|" + ":--:|" * 9]
            ref, best = R.get(REF[optim]), R.get(BEST[optim])
            for m, r in R.items():
                pm = f"{-pct(r['peak'], ref['peak']):+.1f}%" if ref and m != REF[optim] else "—"; sm = f"{x_(r['step_ms'], ref['step_ms']):.2f}×" if ref and m != REF[optim] else "—"
                pb = f"{-pct(r['peak'], best['peak']):+.1f}%" if best and m != BEST[optim] else "—"; sb = f"{x_(r['step_ms'], best['step_ms']):.2f}×" if best and m != BEST[optim] else "—"
                pm_ = f"{r['step_ms']:.1f} ± {r['spread'] if r['seeds'] > 1 else r['step_std']:.1f}"
                out.append(f"| {LABELS[m]} | {r['peak']:.2f} | {pm_} | {r['tflops']:.1f} | {r['tflops'] / PEAK_TFLOPS * 100:.0f}% | {pm} | {sm} | {pb} | {sb} | {tail(r['losses']):.4f} |")
            out += ["", "Every seed :", "", "| Method | Seed | Steps | Peak (GiB) | weights | states | grads | activations | Step (ms) | TF/s | Final loss |", "|---|" + ":--:|" * 10]
            for m, seeds in S.items():
                for r in seeds:
                    g = r["memory_gib"]
                    out.append(f"| {LABELS[m]} | {r['seed']} | {len(r['loss'])} | {g['peak']:.3f} | {g['weights']:.3f} | {g['states']:.3f} | {g['grads']:.3f} | {g['activations']:.3f} | "
                               f"{r['step_ms']['median']:.1f} ± {r['step_ms']['std']:.1f} | {r['tflops']['median']:.1f} | {tail(r['loss']):.4f} |")
            P = profiles(optim, model)
            if P:
                out += ["", f"Where the time of one step goes (`profiling.py`, median of {next(iter(P.values()))['steps']} steps, ms) :", "",
                        "| Method | Forward | Backward | Optimizer step | Sum (sync per phase) | Step (one sync) | Python done issuing | FORGE Python | FORGE updates |", "|---|" + ":--:|" * 8]
                for m, p in P.items():
                    s, o = p["per_phase"], p["one_sync"]
                    out.append(f"| {LABELS[m]} | {s['forward']['done']:.1f} | {s['backward']['done']:.1f} | {s['step']['done']:.1f} | {sum(s[q]['done'] for q in s):.1f} | {o['done']:.1f} | {o['step']:.1f} | "
                               f"{o['forge_python']:.1f} | {p['forge_updates'] or '—'} |")
            out.append("")
        if os.path.exists(tests_path(optim)):
            head = open(tests_path(optim)).readline().strip("# \n")
            fails = [(c, " | ".join(k)) for c, *k, _, _, _ in (l.strip().strip("|").split(" | ") for l in open(tests_path(optim)) if "**FAIL**" in l)]
            out += [f"## Correctness · {optim}", "", f"{head} ([tests.md]({os.path.relpath(tests_path(optim), RESULTS)})). Fails : " + ("; ".join(f"{c.strip()} : {k.strip()}" for c, k in fails) or "none"), ""]
    for optim in ("adamw", "muon"):
        R = big(optim)
        if not R: continue
        r0, ref, best = next(iter(R.values())), R.get(REF[optim]), R.get(BEST[optim])
        K = json.load(open(kaggle_path(optim, next(iter(R)))))["one_device"]["K"]
        out += [f"## Llama-3.1-8B · {optim} · TPU v5e-8", "", f"8 chips, tensor parallel (benchmarks/kaggle/), batch 1, sequence {r0['settings']['seq']}, one seed; step time : median of {r0['steps'] - r0['warmup']} steps "
                f"(the first {r0['warmup']} not timed, lr warmup {r0['settings']['lr_warmup']}). Peak per chip : measured. Peak of one device : CALCULATED from it "
                f"(weights, states, weight gradient x 8, activations x {K}; benchmarks/kaggle/one_device.py).", "",
                "| Method | Peak per chip, measured (GiB) | Peak of one device, calculated (GiB) | weights | states | grads | activations | Step (ms) | TF/s (8 chips) | One device vs. vanilla | vs. best torch | Step vs. vanilla | vs. best torch |", "|---|" + ":--:|" * 13]
        for m, r in R.items():
            cmp = lambda b, k: "—" if b is None or b is r else (f"{-pct(r[k], b[k]):+.1f}%" if k == "peak" else f"{x_(r[k], b[k]):.2f}×")
            out.append(f"| {LABELS[m]} | {r['chip_peak']:.2f} | {r['peak']:.2f} | {r['weights']:.2f} | {r['states']:.2f} | {r['grads']:.2f} | {r['activations']:.2f} | {r['step_ms']:.1f} ± {r['step_std']:.1f} | "
                       f"{r['tflops']:.0f} | {cmp(ref, 'peak')} | {cmp(best, 'peak')} | {cmp(ref, 'step_ms')} | {cmp(best, 'step_ms')} |")
        k_root = os.path.join(RESULTS, "kaggle_v5e8", optim)
        for f, what in (("smoke.md", "8-chip checks"), ("tests.md", "correctness tests, one chip")):
            if os.path.exists(os.path.join(k_root, f)):
                text = open(os.path.join(k_root, f)).read()
                verdict = ("all pass" if "ALL PASS" in text else "some fail") if f == "smoke.md" else text.splitlines()[0].strip("# ")
                out.append(f"\n{what} : {verdict} ([{f}](kaggle_v5e8/{optim}/{f}))")
        out.append("")
    path = os.path.join(RESULTS, "summary.md"); open(path, "w").write("\n".join(out)); print(f"wrote {path}")

def readme(V, T):
    # README.template.md -> README.md : {{key}} or {{key:format}} -> the value. An unknown key stops it
    def fill(match):
        key, fmt = match.group(1), match.group(2)
        v = {**V, **T}[key]                                                                  # KeyError : the template asks for a number that does not exist
        return format(v, fmt) if fmt else str(v)
    text = re.sub(r"\{\{\s*([\w.]+)(?::([^}]+))?\s*\}\}", fill, open(os.path.join(ROOT, "benchmarks", "README.template.md")).read())
    open(os.path.join(ROOT, "README.md"), "w").write(text); print("wrote README.md")



#----------------------------------------------------------------------------------------------- main -------------------------------------------------------------------------------------------------#



def main():
    shutil.rmtree(ASSETS, ignore_errors = True); os.makedirs(ASSETS)
    tiling(os.path.join(ASSETS, "forge_tiling.png"))
    rows = []
    for model, title in MODELS.items():
        R = load("adamw", model)
        if len(R) < 2: continue
        headline(R, "adamw", os.path.join(ASSETS, f"{model}_headline.png"), f"(a) {title} : peak memory")
        rows.append((title, {"mem":   {c: pct(R[m]["peak"], R["adamw"]["peak"]) for c, m in (("f", "adamw_fused"), ("g", "forge_bf16"), ("i", "forge_int8")) if m in R},
                             "speed": {c: x_(R[m]["step_ms"], R["adamw"]["step_ms"]) for c, m in (("f", "adamw_fused"), ("g", "forge_bf16"), ("i", "forge_int8")) if m in R}}))
        if model == "llama": gif(R, "adamw", f"{title}, one TPU v5e chip", os.path.join(ASSETS, "convergence.gif"))
    if rows: glance(rows, os.path.join(ASSETS, "glance.png"))
    R = big("adamw")
    if len(R) > 1: headline(R, "adamw", os.path.join(ASSETS, "llama8b_headline.png"), "(a) Llama-3.1-8B : one device (calculated)")
    summary()
    V, D, gaps, fails = values()
    wrong = [c for ok, c in claims(V, D, gaps, fails) if not ok]
    if wrong: print("README.md NOT written, these claims are false now :\n  " + "\n  ".join(wrong))
    else: readme(V, tables(V, D))
    print("assets :", " ".join(sorted(os.listdir(ASSETS))))

if __name__ == "__main__": main()
