import glob, json, os, re, shutil, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import METHODS, BEST, LABELS, MODELS, RESULTS, ASSETS, run_path, profile_path, tests_path

#   Readme.md
#     python benchmarks/report.py   ->   results/summary.md (every table) and assets/ (every figure). Plain Python + matplotlib : no torch, no TPU, run it anywhere.
#     It reads only what train.py, profiling.py and tests.py wrote (config.py says where), so every number it prints and draws is a recorded one. Several seeds of one
#     method : the numbers are the mean over the seeds, "±" is half the gap between the seeds, the losses are the per-step mean (the band in the plots : min to max).
#        assets/forge_tiling.png                 the kernel, drawn (no data)
#        assets/<model>_headline.png             AdamW : peak memory, split in weights / optimizer states / activations / weight gradient, and memory vs. speed
#        assets/<model>_muon_headline.png        the same for Muon
#        assets/glance.png                       every model at once against vanilla AdamW : memory saved, speed-up
#        assets/<model>_parity.png, assets/<model>_muon_parity.png, assets/convergence.gif     the losses, step by step
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

def parity(R, optim, title, path, note = ""):
    # the parity plot : raw losses (thin, light), smoothed losses (bold), the seed range (band, when there are several seeds), a zoom on the last 15 %, the gap to vanilla torch
    plt = plt_()
    n = min(r["steps"] for r in R.values()); s = span_of(n)
    order = [m for m in ORDER[optim] if m in R]
    fig, ax = plt.subplots(figsize = (7.2, 5.0))
    for m in order: ax.plot(range(n), R[m]["losses"][:n], color = "#a99aa8", lw = 0.6, alpha = 0.45, zorder = 1)
    for m in order:
        c, ls = LINE[m]
        if "lo" in R[m] and R[m].get("seeds", 1) > 1: ax.fill_between(range(n), smooth(R[m]["lo"][:n], s), smooth(R[m]["hi"][:n], s), color = c, alpha = 0.10, lw = 0, zorder = 2)
        ax.plot(range(n), smooth(R[m]["losses"][:n], s), color = c, ls = ls, lw = 2.1 if m.startswith("forge") else 1.7, label = LABELS[m], zorder = 3)
    hi = max(sorted(R[m]["losses"][:n])[int(n * 0.99) - 1] for m in order)
    ax.set_ylim(0, hi * 1.06); ax.set_xlabel("training step", fontsize = 11); ax.set_ylabel("training loss (nats)" if "MLP" not in title else "training loss (MSE)", fontsize = 11)
    ax.set_title(title, fontsize = 12); ax.grid(alpha = 0.35); ax.legend(fontsize = 8.5, loc = "upper right", framealpha = 0.95)
    for sp in ("top", "right"): ax.spines[sp].set_visible(True)
    z0 = int(n * 0.85)
    end = statistics.mean(tail(R[m]["losses"][:n]) for m in order)
    ins = ax.inset_axes([0.33, 0.42, 0.36, 0.40] if end < 0.3 * hi else [0.40, 0.07, 0.36, 0.36])        # the zoom where the curves are not : top middle, or bottom when they end high
    for m in order:
        c, ls = LINE[m]
        ins.plot(range(z0, n), smooth(R[m]["losses"][:n], s)[z0:], color = c, ls = ls, lw = 1.5)
    zs = [v for m in order for v in smooth(R[m]["losses"][:n], s)[z0:]]
    pad = (max(zs) - min(zs)) * 0.25 + 1e-4
    ins.set_ylim(min(zs) - pad, max(zs) + pad); ins.tick_params(labelsize = 6.5); ins.set_title(f"zoom: steps {z0}–{n}", fontsize = 7.5); ins.set_facecolor("white")
    ref = REF[optim]
    if ref in R:
        gaps = {m: abs(tail(R[m]["losses"][:n]) - tail(R[ref]["losses"][:n])) for m in order if m != ref}
        fig.text(0.5, 0.012, f"final-loss gap to {LABELS[ref]} (nats, mean of the last 10 % of steps){note}:    " + ",    ".join(f"{LABELS[m]}: {v:.4f}" for m, v in gaps.items()), fontsize = 7, ha = "center", va = "bottom")
    fig.tight_layout(rect = (0, 0.04, 1, 1)); fig.savefig(path, dpi = 150); plt.close(fig)

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



#---------------------------------------------------------------------------------------------- tables ------------------------------------------------------------------------------------------------#



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
    path = os.path.join(RESULTS, "summary.md"); open(path, "w").write("\n".join(out)); print(f"wrote {path}")



#----------------------------------------------------------------------------------------------- main -------------------------------------------------------------------------------------------------#



def main():
    shutil.rmtree(ASSETS, ignore_errors = True); os.makedirs(ASSETS)
    tiling(os.path.join(ASSETS, "forge_tiling.png"))
    rows = []
    for model, title in MODELS.items():
        R = load("adamw", model)
        if len(R) > 1:
            seeds = f", {R['adamw']['seeds']} seeds" if R.get("adamw", {}).get("seeds", 1) > 1 else ""
            headline(R, "adamw", os.path.join(ASSETS, f"{model}_headline.png"), f"(a) {title} : peak memory")
            parity(R, "adamw", f"{title} (AdamW, one TPU v5e{seeds})", os.path.join(ASSETS, f"{model}_parity.png"))
            if "adamw" in R: rows.append((title, {"mem": {c: pct(R[m]["peak"], R["adamw"]["peak"]) for c, m in (("f", "adamw_fused"), ("g", "forge_bf16"), ("i", "forge_int8")) if m in R},
                                                  "speed": {c: x_(R[m]["step_ms"], R["adamw"]["step_ms"]) for c, m in (("f", "adamw_fused"), ("g", "forge_bf16"), ("i", "forge_int8")) if m in R}}))
            if model == "llama": gif(R, "adamw", f"{title}, one TPU v5e chip", os.path.join(ASSETS, "convergence.gif"))
        R = load("muon", model)
        if len(R) > 1:
            headline(R, "muon", os.path.join(ASSETS, f"{model}_muon_headline.png"), f"(a) {title}, Muon : peak memory")
            parity(R, "muon", f"{title} (Muon, one TPU v5e)", os.path.join(ASSETS, f"{model}_muon_parity.png"))
    if rows: glance(rows, os.path.join(ASSETS, "glance.png"))
    summary()
    print("assets :", " ".join(sorted(os.listdir(ASSETS))))

if __name__ == "__main__": main()
