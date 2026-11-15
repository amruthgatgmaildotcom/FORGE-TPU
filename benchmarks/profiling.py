import importlib, json, os, statistics, sys, time, torch, warnings
from forge_tpu import _bridge
from mlp import OPTIMIZERS, LABELS

# Where does the time of one training step go ? For one model and one method (one process each, as the benchmarks), on the same model, data and optimizer as <model>.py.
# Two ways to run the same step, each WARMUP + STEPS times (median) :
#   "as benchmark" : forward, backward, step, then ONE sync (as <model>.py times it). Python's clock after each phase = when Python has ISSUED that phase,
#                    the sync = when the TPU has FINISHED all of it. The TPU works while Python issues the next ops : this is the overlap
#   "sync per phase" : a sync after each phase. For each phase : how long Python takes to issue it, and when the TPU has finished it (no overlap between phases)
# Then PROFILED steps under torch.profiler (CPU side only : the TPU is not seen by it) : Python / host time of each phase, in a few groups (matmul, attention, ...).
# The profiler slows the host, so the groups show where host time goes, not the real step time.
WARMUP, STEPS, PROFILED = 10, 20, 3
OUT = os.environ.get("RESULTS", "results")
PHASES = ("forward", "backward", "step")
GROUPS = {                                                             # host time groups : the first group whose pattern is in the op name (lower case) takes it
    "FORGE update (Python)": ("fusedlinearfunction",),
    "optimizer":             ("optimizer.step", "fused_adam", "_foreach", "addcdiv", "addcmul", "lerp"),
    "matmul":                ("aten::mm", "aten::addmm", "aten::bmm", "aten::matmul", "aten::linear"),
    "attention":             ("scaled_dot_product", "attention"),
    "autograd engine":       ("autograd::", "backward0", "backward1"),
    "copies, new tensors":   ("empty", "zeros", "ones", "full", "fill_", "zero_", "copy_", "_to_copy", "aten::to", "clone", "contiguous", "aten::cat", "embedding", "index", "_backward"),
    "views (no data moved)": ("view", "reshape", "transpose", "permute", "expand", "squeeze", "aten::t", "slice", "select", "as_strided", "chunk", "split", "alias", "detach", "narrow"),
    "elementwise":           ("aten::",),                              # every other aten op : mul, add, neg, pow, silu, gelu, rsqrt, sum, mean, ...
}
OTHER = "Python, waiting (no op)"                                      # phase time not inside any op : Python code of the model, autograd waiting for the TPU, ...


#------------------------------------------------------------------------- measure -------------------------------------------------------------------------#

def group(name):
    name = name.lower()
    return next((g for g, keys in GROUPS.items() if any(k in name for k in keys)), OTHER)

def measure(prefix, name):
    model_file = importlib.import_module(prefix)                       # mlp, llama or qwen : the same make_model, make_data, loss_fn as the benchmark
    sync = torch.accelerator.synchronize
    data = model_file.make_data()
    model, opt = model_file.make_model(name); model.train()
    forge = [0.0, 0]                                                   # Python time inside FORGE's fused updates (seconds), number of updates
    original = _bridge.fused_update
    def timed(*args, **kwargs):
        t = time.perf_counter(); out = original(*args, **kwargs); forge[0] += time.perf_counter() - t; forge[1] += 1; return out
    _bridge.fused_update = timed                                       # autograd.py calls _bridge.fused_update at run time, so it calls this one

    def as_benchmark(x, y, mark = lambda phase: None):
        # returns the time (ms from the start) when Python has issued forward, backward, step, and when the TPU is done
        sync(); forge[0] = 0.0; t0 = time.perf_counter(); t = {}
        mark("forward"); loss = model_file.loss_fn(model(x), y); t["forward"] = time.perf_counter()
        mark("backward"); loss.backward(); t["backward"] = time.perf_counter()
        mark("step")
        if opt: opt.step(); opt.zero_grad(set_to_none = True)
        t["step"] = time.perf_counter(); mark(None); sync(); t["done"] = time.perf_counter()
        return {k: (v - t0) * 1e3 for k, v in t.items()} | {"forge python": forge[0] * 1e3}

    def sync_per_phase(x, y):
        # for each phase : ms until Python has issued it ("host") and ms until the TPU has finished it ("done"), both from the start of that phase
        r = {}
        sync(); t0 = time.perf_counter(); loss = model_file.loss_fn(model(x), y); t1 = time.perf_counter(); sync(); t2 = time.perf_counter()
        r["forward"] = {"host": (t1 - t0) * 1e3, "done": (t2 - t0) * 1e3}
        t0 = time.perf_counter(); loss.backward(); t1 = time.perf_counter(); sync(); t2 = time.perf_counter()
        r["backward"] = {"host": (t1 - t0) * 1e3, "done": (t2 - t0) * 1e3}
        t0 = time.perf_counter()
        if opt: opt.step(); opt.zero_grad(set_to_none = True)
        t1 = time.perf_counter(); sync(); t2 = time.perf_counter()
        r["step"] = {"host": (t1 - t0) * 1e3, "done": (t2 - t0) * 1e3}
        return r

    batch = lambda step: data[step % len(data)]
    for step in range(WARMUP): as_benchmark(*batch(step))              # compile
    A = [as_benchmark(*batch(step)) for step in range(STEPS)]
    B = [sync_per_phase(*batch(step)) for step in range(STEPS)]
    overlapped = {k: statistics.median(a[k] for a in A) for k in A[0]}
    split = {p: {k: statistics.median(b[p][k] for b in B) for k in ("host", "done")} for p in PHASES}

    # profiled steps : host time of each phase, by group. A phase = the time between two record_function marks (any thread : the backward can run on another one)
    from torch.profiler import profile, ProfilerActivity, record_function
    host = {p: {} for p in PHASES}
    with profile(activities = [ProfilerActivity.CPU]) as prof:
        for step in range(PROFILED):
            current = []
            def mark(phase):
                if current: current.pop().__exit__(None, None, None)
                if phase: current.append(record_function(f"phase: {phase}")); current[-1].__enter__()
            as_benchmark(*batch(step), mark)
    events = list(prof.events())
    spans = [(e.name[7:], e.time_range.start, e.time_range.end) for e in events if e.name.startswith("phase: ")]
    for e in events:
        if e.name.startswith("phase: "): continue
        phase = next((p for p, a, b in spans if a <= e.time_range.start < b), None)
        if phase: host[phase][group(e.name)] = host[phase].get(group(e.name), 0) + e.self_cpu_time_total / 1e3 / PROFILED
    for p in PHASES:                                                   # what is not in an op : phase length - time in ops (the sum of several threads can be more : then 0)
        length = sum(b - a for q, a, b in spans if q == p) / 1e3 / PROFILED
        host[p][OTHER] = max(length - sum(host[p].values()), 0.0)
    R = dict(model = prefix, name = name, overlapped = overlapped, split = split, host = host, forge_updates = forge[1] // (WARMUP + 2 * STEPS + PROFILED) if forge[1] else 0,
             replayed_keys = len(_bridge._LAUNCHES), keys_on_jax_op = len(_bridge._NO_REPLAY), profiler_activities = sorted(str(a) for a in torch.profiler.supported_activities()))
    print(summary(R), flush = True)
    return R


#------------------------------------------------------------------------- report --------------------------------------------------------------------------#

PHASE_COLORS = {"forward": "#2f7ed8", "backward": "#e5484d", "step": "#14896b"}
WAIT = "#b8bcc6"
GROUP_COLORS = dict(zip([*GROUPS, OTHER], ("#14896b", "#8c5bd6", "#2f7ed8", "#f2a93b", "#e5484d", "#7fb8e6", "#d9dce3", "#f7c6c7", "#8c8c8c")))

def summary(R):
    o, s = R["overlapped"], R["split"]
    lines = [f"{R['model']} {LABELS[R['name']]} : step {o['done']:.1f} ms as benchmark (Python done issuing at {o['step']:.1f} ms)",
             "  sync per phase : " + "   ".join(f"{p} {s[p]['host']:.1f} Python / {s[p]['done']:.1f} TPU done" for p in PHASES) + f"   (sum {sum(s[p]['done'] for p in PHASES):.1f} ms)"]
    if R["forge_updates"]: lines.append(f"  FORGE : {R['forge_updates']} fused updates per step, {o['forge python']:.1f} ms of Python in them, replayed launch keys {R['replayed_keys']}, keys on the slow jax_op path {R['keys_on_jax_op']}")
    return "\n".join(lines)

def timeline(P, title, path):
    # one step on a time axis, two lanes per method. "as benchmark" : when Python issues each phase, then (grey) Python waits for the TPU to finish.
    # "sync per phase" : each phase alone. dark = Python issuing (the TPU works at the same time), light = only the TPU works, Python waits
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    names = list(P); fig, ax = plt.subplots(figsize = (15, 1.1 + 1.25 * len(names)))
    ticks, labels = [], []
    for i, n in enumerate(names):
        o, s, y = P[n]["overlapped"], P[n]["split"], 2.6 * (len(names) - 1 - i)
        start = 0.0
        for p in PHASES:
            ax.broken_barh([(start, o[p] - start)], (y + 1.25, 0.9), color = PHASE_COLORS[p]); start = o[p]
        ax.broken_barh([(start, o["done"] - start)], (y + 1.25, 0.9), color = WAIT)
        ax.text(o["done"] + 1, y + 1.7, f"{o['done']:.1f} ms", va = "center", fontsize = 10, fontweight = "bold")
        if o["forge python"]: ax.text(o["forward"] + 1, y + 1.7, f"FORGE Python {o['forge python']:.0f} ms (spread in it)", va = "center", fontsize = 8, color = "white")
        start = 0.0
        for p in PHASES:
            ax.broken_barh([(start, s[p]["host"])], (y, 0.9), color = PHASE_COLORS[p])
            ax.broken_barh([(start + s[p]["host"], s[p]["done"] - s[p]["host"])], (y, 0.9), color = PHASE_COLORS[p], alpha = 0.3)
            start += s[p]["done"]
        ax.text(start + 1, y + 0.45, f"{start:.1f} ms", va = "center", fontsize = 10)
        ticks += [y + 1.7, y + 0.45]; labels += [f"{LABELS[n]}   as benchmark", "sync per phase"]
    ax.set_yticks(ticks, labels, fontsize = 11); ax.set_xlabel("Time in one training step (ms)", fontsize = 12); ax.grid(axis = "x", ls = ":")
    ax.set_xlim(0, max(max(P[n]["overlapped"]["done"], sum(P[n]["split"][p]["done"] for p in PHASES)) for n in names) * 1.12)
    ax.legend(handles = [Patch(color = PHASE_COLORS[p], label = f"{p} : Python issues ops (TPU works too)") for p in PHASES] +
              [Patch(color = PHASE_COLORS["backward"], alpha = 0.3, label = "light : only the TPU works, Python waits"), Patch(color = WAIT, label = "grey : Python is done, the TPU finishes the step")],
              fontsize = 9, ncol = 3, loc = "upper center", bbox_to_anchor = (0.5, -0.14), frameon = False)
    ax.set_title(f"{title} : where the time of one step goes", fontsize = 15, fontweight = "bold", loc = "left"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi = 200); plt.close(fig)

def host_groups(P, title, path):
    # host (Python + PyTorch) time of each phase, in groups, one stacked bar per method. Under the profiler (slower than a real step)
    import matplotlib.pyplot as plt
    names = list(P); fig, axes = plt.subplots(1, 3, figsize = (17, 1.6 + 0.6 * len(names)), sharey = True)
    for ax, p in zip(axes, PHASES):
        for i, n in enumerate(names):
            left = 0.0
            for g in [*GROUPS, OTHER]:
                w = P[n]["host"][p].get(g, 0.0)
                ax.barh(len(names) - 1 - i, w, left = left, color = GROUP_COLORS[g], label = g if i == 0 else None, height = 0.7); left += w
            ax.text(left, len(names) - 1 - i, f" {left:.1f}", va = "center", fontsize = 9)
        ax.set_title(p, fontsize = 13, fontweight = "bold", color = PHASE_COLORS[p]); ax.set_xlabel("host time (ms, under the profiler)"); ax.grid(axis = "x", ls = ":")
        ax.spines[["top", "right"]].set_visible(False); ax.set_xlim(0, max(sum(P[n]["host"][p].values()) for n in names) * 1.25 + 1e-3)
    axes[0].set_yticks(range(len(names)), [LABELS[n] for n in reversed(names)], fontsize = 11)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol = 5, fontsize = 9, loc = "lower center", frameon = False)
    fig.suptitle(f"{title} : what Python / PyTorch does on the host in each phase", x = 0.01, ha = "left", fontsize = 14, fontweight = "bold")
    fig.tight_layout(rect = (0, 0.1, 1, 1)); fig.savefig(path, dpi = 200); plt.close(fig)

def simple(P, title):
    # the short form : one row per phase, one column per method (ms)
    names = list(P); o = {n: P[n]["overlapped"] for n in names}; s = {n: P[n]["split"] for n in names}
    row = lambda label, f: f"| {label} | " + " | ".join(f(n) for n in names) + " |"
    rows = [f"### {title} : where the time of one step goes (ms, median of {STEPS} steps)", "", "| | " + " | ".join(LABELS[n] for n in names) + " |", "|---|" + ":--:|" * len(names),
            "| **Sync after each phase** |" + " |" * len(names),
            row("Forward", lambda n: f"{s[n]['forward']['done']:.1f}"),
            row("Backward : Python returns", lambda n: f"{s[n]['backward']['host']:.1f}"),
            row("Backward : TPU finishes", lambda n: f"{s[n]['backward']['done']:.1f}"),
            row("Optimizer step", lambda n: f"{s[n]['step']['done']:.1f}"),
            row("Total", lambda n: f"{sum(s[n][p]['done'] for p in PHASES):.1f}"),
            "| **As the benchmark (one sync at the end)** |" + " |" * len(names),
            row("Python done issuing", lambda n: f"{o[n]['step']:.1f}"),
            row("Step time (TPU done)", lambda n: f"**{o[n]['done']:.1f}**"),
            row("of which FORGE Python", lambda n: f"{o[n]['forge python']:.1f}" if P[n]["forge_updates"] else "-"),
            row("FORGE updates per step", lambda n: str(P[n]["forge_updates"]) if P[n]["forge_updates"] else "-")]
    return "\n".join(rows)

def table(P, title):
    rows = [simple(P, title), "", f"### {title} : profile in detail (median of {STEPS} steps, after {WARMUP} warmup steps)", "",
            "| Method | Step as benchmark (ms) | Python done issuing (ms) | Forward : Python / TPU done | Backward : Python / TPU done | Step : Python / TPU done | FORGE Python (ms) |",
            "|---|:--:|:--:|:--:|:--:|:--:|:--:|"]
    for n, r in P.items():
        o, s = r["overlapped"], r["split"]
        rows.append(f"| {LABELS[n]} | {o['done']:.1f} | {o['step']:.1f} | " + " | ".join(f"{s[p]['host']:.1f} / {s[p]['done']:.1f}" for p in PHASES) + f" | {o['forge python']:.1f} |")
    rows += ["", "Host time per phase (ms, under the profiler), the 3 largest groups :", "", "| Method | Forward | Backward | Step |", "|---|---|---|---|"]
    top = lambda d: ", ".join(f"{g} {v:.1f}" for g, v in sorted(d.items(), key = lambda kv: -kv[1])[:3])
    rows += [f"| {LABELS[n]} | " + " | ".join(top(r["host"][p]) for p in PHASES) + " |" for n, r in P.items()]
    return "\n".join(rows)

def report(prefix):
    P = {n: json.load(open(f"{OUT}/profile_{prefix}_{n}.json")) for n in OPTIMIZERS if os.path.exists(f"{OUT}/profile_{prefix}_{n}.json")}
    title = {"mlp": "MLP", "llama": "Llama-3.2-1B", "qwen": "Qwen3-0.6B"}.get(prefix, prefix)
    timeline(P, title, f"{OUT}/profile_{prefix}_timeline.png"); host_groups(P, title, f"{OUT}/profile_{prefix}_host.png")
    md = table(P, title); open(f"{OUT}/profile_{prefix}_table.md", "w").write(md + "\n")
    print("\n\n".join(summary(r) for r in P.values()) + "\n\n" + md)


if __name__ == "__main__":
    # python profiling.py <mlp | llama | qwen> <method> : profile one method (one process each, so memory is clean)   |   python profiling.py <model> report : plots + table
    warnings.filterwarnings("ignore"); os.makedirs(OUT, exist_ok = True)
    prefix, what = sys.argv[1], sys.argv[2]
    if what == "report": report(prefix)
    else: json.dump(measure(prefix, what), open(f"{OUT}/profile_{prefix}_{what}.json", "w"))
