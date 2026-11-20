# FORGE TPU : the whole benchmark in ONE Colab cell (TPU v5e-1). Paste this file into a cell, run it, upload the repo zip when asked (GitHub : Code > Download ZIP).
# Every run is its own process (the TPU takes one process at a time, and the memory of one run is never mixed with another). The order is the order of importance :
# if the Colab time ends, interrupt the cell (Runtime > Interrupt) : results.zip of everything done so far is still downloaded.
#   1. tests      benchmarks/tests.py        for every optimizer of TESTS                       -> results/<optim>/tests.md
#   2. train      benchmarks/train.py        every method of every optimizer of TRAIN, seeds    -> results/<optim>/<model>/<method>_seed<N>.json
#   3. profile    benchmarks/profiling.py    every method of every optimizer of PROFILE         -> results/<optim>/<model>/profile_<method>.json
#   4. report     benchmarks/report.py       tables and figures from everything above           -> results/summary.md, assets/
import json, os, re, shutil, subprocess, sys, time, zipfile
from IPython.display import Image, Markdown, display
from google.colab import files

TESTS   = ["adamw"]                                                                          # optimizers
TRAIN   = {"adamw": ["mlp", "llama", "qwen"], "muon": ["llama", "qwen"]}                    # optimizer : models
PROFILE = {"adamw": ["llama", "qwen"]}                                                       # optimizer : models
SEEDS   = {"adamw": [0, 1], "muon": [0]}                                                       # optimizer : seeds. Every first seed runs before any second one
ENV     = {"SEQ": "512", "MAX_SECONDS": "120"}                                               # see benchmarks/config.py

QUIET = {"PYTHONWARNINGS": "ignore", "TF_CPP_MIN_LOG_LEVEL": "3", "TPU_STDERR_LOG_LEVEL": "3", "TPU_MIN_LOG_LEVEL": "3", "GLOG_minloglevel": "3", "GRPC_VERBOSITY": "ERROR"}
NOISE = re.compile(r"^[IW]\d{4} |^WARNING: All log messages|^Successfully renamed PrivateUse1|^Registered Python module|^Device type:|^Initializing TPU|^num_cache|^num_compilations|^peak_compilation|Warning: ")
REPO, START, FAILED = "/content/TRAVELLER", time.time(), []

def run(*args, env = None):
    # a child process, its output shown here without the TPU runtime noise. Returns its exit code
    p = subprocess.Popen([sys.executable, "-u", *args], cwd = REPO if os.path.isdir(REPO) else None, env = {**os.environ, **QUIET, **(env or {})}, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True)
    for line in p.stdout:
        if not NOISE.search(line): print(line, end = "")
    return p.wait()

def step(what, *args, env):
    print(f"\n[{int(time.time() - START) // 60:3d} min] ===== {what} =====")
    if run(*args, env = {**BASE, **env}) != 0: FAILED.append(what)

# 0. the TPU, transformers, the repo zip
assert run("-c", "import torch; x = torch.ones(1024, 1024, device = 'tpu', dtype = torch.bfloat16); assert (x @ x)[0, 0].item() == 1024; print('TPU OK')") == 0, "STOP : no TPU (Runtime > Change runtime type > TPU)"
if run("-c", "from transformers import LlamaForCausalLM, Qwen3ForCausalLM") != 0: assert run("-m", "pip", "install", "-q", "-U", "transformers") == 0, "STOP : transformers cannot be installed"
shutil.rmtree(REPO, ignore_errors = True)
zipfile.ZipFile(os.path.abspath(next(iter(files.upload())))).extractall(REPO)
inner = [n for n in os.listdir(REPO) if n != "__MACOSX" and not n.startswith(".")]
if not os.path.exists(f"{REPO}/src") and len(inner) == 1:                                    # a zip with one top folder : move its content up
    for n in os.listdir(f"{REPO}/{inner[0]}"): shutil.move(f"{REPO}/{inner[0]}/{n}", f"{REPO}/{n}")
    os.rmdir(f"{REPO}/{inner[0]}")
assert os.path.exists(f"{REPO}/benchmarks/train.py") and os.path.exists(f"{REPO}/src/forge_tpu/__init__.py"), "STOP : the zip must hold src/ and benchmarks/"
shutil.rmtree(f"{REPO}/results", ignore_errors = True); shutil.rmtree(f"{REPO}/assets", ignore_errors = True)   # only the results of this run
BASE = {"PYTHONPATH": f"{REPO}/src:{REPO}/benchmarks", **ENV}
METHODS = json.loads(subprocess.run([sys.executable, "-c", "import json, config; print(json.dumps(config.METHODS))"], cwd = f"{REPO}/benchmarks", capture_output = True, text = True).stdout)   # config.py : plain Python

try:
    for optim in TESTS: step(f"tests {optim}", "benchmarks/tests.py", env = {"OPTIM": optim})
    for i in range(max(len(seeds) for seeds in SEEDS.values())):
        for optim, models in TRAIN.items():
            for seed in SEEDS[optim][i:i + 1]:
                for model in models:
                    for method in METHODS[optim]: step(f"train {optim} {model} {method} seed {seed}", "benchmarks/train.py", model, method, env = {"OPTIM": optim, "SEED": str(seed)})
        if i == 0:                                                                           # the profiles after the first seeds : they matter more than a second seed
            for optim, models in PROFILE.items():
                for model in models:
                    for method in METHODS[optim]: step(f"profile {optim} {model} {method}", "benchmarks/profiling.py", model, method, env = {"OPTIM": optim})
finally:
    step("report", "benchmarks/report.py", env = {})
    if os.path.exists(f"{REPO}/results/summary.md"): display(Markdown(open(f"{REPO}/results/summary.md").read()))
    for f in sorted(os.listdir(f"{REPO}/assets")) if os.path.isdir(f"{REPO}/assets") else []:
        if f.endswith(".png"): display(Image(f"{REPO}/assets/{f}"))
    print("\nFAILED : " + ", ".join(FAILED) if FAILED else "\nevery run finished")
    with zipfile.ZipFile("/content/results.zip", "w", zipfile.ZIP_DEFLATED) as z:                # results/ and assets/, as they go into the repo
        for top in ("results", "assets"):
            for root, _, fs in os.walk(f"{REPO}/{top}"):
                for f in fs: z.write(f"{root}/{f}", os.path.relpath(f"{root}/{f}", REPO))
    files.download("/content/results.zip")
