import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from methods import METHODS, LABELS

#   Readme.md
#     Writes the numbers and plots of results/ into the READMEs, so no README holds a number by hand. Plain Python : no torch, no TPU.
#     python benchmarks/readme.py (from the repo root, or anywhere) does two things :
#        1. results/<optim>/README.md for every results/<optim>/ folder : written fully, results only (tables + plots of each model + its profile).
#        2. README.md (the repo root) : only the blocks between <!-- auto:KIND ARGS --> and <!-- /auto --> are written again, the text around them is kept.
#           KIND results  ARGS optim/model                 : the benchmark table, the loss table and the three plots of that model
#           KIND profile  ARGS optim/model                 : the profile tables and the two profile plots of that model
#           KIND lossdiff ARGS optim/model method_a method_b : max |loss of a - loss of b| over the steps that both runs have (inline number)
#     colab_cell_all.py runs it after the reports, so results.zip has the optimizer READMEs. After the zip is unpacked into results/, run it again for README.md.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
WARMUP = 10                                                            # as mlp.WARMUP : the first steps of a run are not timed
MODELS = {"mlp": "MLP", "llama": "Llama-3.2-1B", "qwen": "Qwen3-0.6B"}



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def runs(optim, model):
    # {method: the json of its run} for the methods of optim that have a result file, in the order of methods.py
    folder = os.path.join(RESULTS, optim, model)
    paths = {n: os.path.join(folder, f"{model}_{n}.json") for n in METHODS[optim]}
    return {n: json.load(open(p)) for n, p in paths.items() if os.path.exists(p)}

def timed(r): return r["steps"] - WARMUP if r["steps"] > WARMUP + 1 else r["steps"]   # as mlp.measure : a very short run uses every step

def caption(optim, model, R):
    steps = sorted({timed(r) for r in R.values()})
    count = f"{steps[0]}" if len(steps) == 1 else f"{steps[0]} to {steps[-1]}"
    return (f"{MODELS[model]}, {R[next(iter(R))]['params'] / 1e9:.2f}B params, {optim}. Step time : median (± std) of {count} timed steps of each run, "
            f"after {WARMUP} warmup steps. Peak : the largest step peak above the data.")

def results_block(optim, model, prefix = ""):
    # prefix : the path from the README to results/<optim>/ (the root README : "results/<optim>/", the optimizer README : "")
    R = runs(optim, model)
    if not R: return f"NA : no results in results/{optim}/{model}/"
    order = METHODS[optim]
    rows = [caption(optim, model, R), "", "| Method | Peak (GiB) | Step (ms) | TF/s |", "|---|:--:|:--:|:--:|"]
    rows += [f"| {LABELS[n]} | {R[n]['peak']:.2f} | {R[n]['step_ms']:.1f} ± {R[n]['step_std']:.1f} | {R[n]['tflops']:.0f} ± {R[n]['tflops_std']:.0f} |" if n in R
             else f"| {LABELS[n]} | NA | NA | NA |" for n in order]
    ref = order[0]                                                     # vanilla torch : the reference of the loss table (as mlp.table)
    if ref in R:
        last = min(r["steps"] for r in R.values())                     # the runs have different lengths : compare them at the last step that every run has
        rows += ["", f"| Method | Loss at step {last} (last common step) | max \\|loss − {LABELS[ref]}\\| (steps 1 to {last}) |", "|---|:--:|:--:|"]
        rows += [f"| {LABELS[n]} | {R[n]['losses'][last - 1]:.4f} | {lossdiff(R[n], R[ref], last):.4f} |" if n in R else f"| {LABELS[n]} | NA | NA |" for n in order]
    rows += ["", *(f"![{MODELS[model]} {kind}]({prefix}{model}/{model}_{kind}.png)" for kind in ("headline", "loss", "memory")
                   if os.path.exists(os.path.join(RESULTS, optim, model, f"{model}_{kind}.png")))]
    return "\n".join(rows)

def profile_block(optim, model, prefix = ""):
    # the tables that profiling.py wrote (profile_<model>_table.md), one heading level down, and its two plots
    path = os.path.join(RESULTS, optim, model, f"profile_{model}_table.md")
    if not os.path.exists(path): return f"NA : no profile in results/{optim}/{model}/"
    text = re.sub(r"^###", "####", open(path).read().strip(), flags = re.M)
    plots = [f"![{MODELS[model]} profile {kind}]({prefix}{model}/profile_{model}_{kind}.png)" for kind in ("timeline", "host")
             if os.path.exists(os.path.join(RESULTS, optim, model, f"profile_{model}_{kind}.png"))]
    return "\n".join([text, "", *plots])

def lossdiff(a, b, n = None):
    n = n or min(len(a["losses"]), len(b["losses"]))
    return max(abs(x - y) for x, y in zip(a["losses"][:n], b["losses"][:n]))



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def optimizer_readme(optim):
    title = {"adamw": "AdamW", "sgd": "SGD"}.get(optim, optim)
    parts = [f"# {title} results (TPU v5e-1)", "", "Written by `benchmarks/readme.py` from the files in this folder. Do not edit by hand.", ""]
    for model in MODELS:
        if not os.path.isdir(os.path.join(RESULTS, optim, model)): continue
        parts += [f"## {MODELS[model]}", "", results_block(optim, model), "", "### Profile", "", profile_block(optim, model), ""]
    open(os.path.join(RESULTS, optim, "README.md"), "w").write("\n".join(parts))

AUTO = re.compile(r"(<!-- auto:(\w+) ([^>]*?) -->)(.*?)(<!-- /auto -->)", re.S)

def root_readme():
    path = os.path.join(ROOT, "README.md")
    if not os.path.exists(path): return
    def fill(match):
        start, kind, args, _, end = match.groups()
        target, *rest = args.split()
        optim, model = target.split("/")
        if kind == "lossdiff":                                         # inline : no new lines around the number
            R = runs(optim, model)
            a, b = rest
            return f"{start}{lossdiff(R[a], R[b]):.4f}{end}" if a in R and b in R else f"{start}NA{end}"
        block = {"results": results_block, "profile": profile_block}[kind](optim, model, f"results/{optim}/")
        return f"{start}\n{block}\n{end}"
    text = open(path).read()
    open(path, "w").write(AUTO.sub(fill, text))



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



if __name__ == "__main__":
    for optim in sorted(METHODS):
        if os.path.isdir(os.path.join(RESULTS, optim)): optimizer_readme(optim); print(f"wrote results/{optim}/README.md")
    root_readme(); print("wrote the auto blocks of README.md")
