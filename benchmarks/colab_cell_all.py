# FORGE TPU : MLP + Llama 1B + Qwen3 0.6B benchmarks, then the profile of the same runs (correctness tests : RUN_CORRECTNESS_TESTS = 1), one Colab cell (TPU v5e-1 / v6e-1). ~30 min per optimizer (one benchmark + one profile per method and model).
# Each run stops after MAX_SECONDS (60 here, 180 in the single-model cells), so the runs fit in the free tier. Results : results.zip with <optim>/mlp/, <optim>/llama/, <optim>/qwen/ (optim : adamw, sgd, see OPTIMS)
import os, re, shutil, subprocess, sys, warnings, zipfile
from IPython.display import Image, Markdown, display
from google.colab import files
warnings.filterwarnings("ignore")

RUN_CORRECTNESS_TESTS, RUN_BENCHMARKS, RUN_PROFILING = 0, 1, 1        # 1 : run that part, 0 : skip it. Tests : benchmarks/tests.py, a few minutes, into results/<optim>/tests
OPTIMS = {"adamw", "SGD"}                                              # the optimizers to run (any case) : each one goes to results/<optim>/... Remove one to skip it

QUIET = {"PYTHONWARNINGS": "ignore", "TF_CPP_MIN_LOG_LEVEL": "3", "TPU_STDERR_LOG_LEVEL": "3", "TPU_MIN_LOG_LEVEL": "3", "GLOG_minloglevel": "3", "GRPC_VERBOSITY": "ERROR"}
NOISE = re.compile(r"^[IW]\d{4} |^WARNING: All log messages|^Successfully renamed PrivateUse1|^Registered Python module|^Device type:|^Initializing TPU|^num_cache|^num_compilations|^peak_compilation|Warning: ")

def run(*args, cwd = None, env = None):
    # child process, output streamed here without the TPU runtime noise. The TPU takes one process at a time, so this notebook never touches it itself
    p = subprocess.Popen([sys.executable, "-u", *args], cwd = cwd, env = {**(env or os.environ), **QUIET}, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True)
    skip = False
    for line in p.stdout:
        if skip and line.startswith((" ", "\t")) and "Error" not in line: skip = False; continue   # the source line printed under a warning
        skip = bool(NOISE.search(line))
        if not skip: print(line, end = "")
    return p.wait()

# 1. TPU check : stop here if torch does not see a TPU
CHECK = r'''
import importlib.metadata as md, os, torch
print("packages :", sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions() if any(k in d.metadata["Name"].lower() for k in ("torch", "jax", "tpu"))))
dev = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else None
assert dev is not None and dev.type == "tpu", f"torch sees no TPU (accelerator = {dev}). Runtime > Change runtime type > TPU"
x = torch.ones(1024, 1024, device = "tpu", dtype = torch.bfloat16)
assert (x @ x)[0, 0].item() == 1024, "TPU matmul gives a wrong value"
print("TPU OK :", dev, os.environ.get("TPU_ACCELERATOR_TYPE", ""))
'''
assert run("-c", CHECK) == 0, "STOP : no working TPU (see the message above)"

# 1b. llama.py and qwen.py take the model code from the transformers library (pure Python, the weights are NOT downloaded) : install it if it is missing or too old for Qwen3
HAVE = "try: from transformers import LlamaForCausalLM, Qwen3ForCausalLM\nexcept Exception: raise SystemExit(1)"
if run("-c", HAVE) != 0: assert run("-m", "pip", "install", "-q", "-U", "transformers") == 0 and run("-c", HAVE) == 0, "STOP : transformers cannot be installed / imported (see the message above)"

# 2. the repo zip -> /content/TRAVELLER. Both forms work : benchmarks/, src/, README.md at the top of the zip, or inside one top folder
REPO = "/content/TRAVELLER"
shutil.rmtree(REPO, ignore_errors = True)
zipfile.ZipFile(os.path.abspath(next(iter(files.upload())))).extractall(REPO)
inner = [name for name in os.listdir(REPO) if name != "__MACOSX" and not name.startswith(".")]   # __MACOSX : a folder that the macOS zip tool adds
if not os.path.exists(f"{REPO}/src") and len(inner) == 1 and os.path.isdir(f"{REPO}/{inner[0]}"):   # a zip with one top folder (GitHub : Code > Download ZIP) : move its content up
    top = f"{REPO}/{inner[0]}"
    for name in os.listdir(top): shutil.move(f"{top}/{name}", f"{REPO}/{name}")
    os.rmdir(top)
shutil.rmtree(f"{REPO}/results", ignore_errors = True)                # results/ of the repo are old runs : a report must never show them in place of a run that failed
assert os.path.exists(f"{REPO}/src/forge_tpu/__init__.py") and os.path.exists(f"{REPO}/benchmarks/qwen.py"), f"STOP : the zip must hold src/ and benchmarks/ (at its top level or in one top folder), it holds {os.listdir(REPO)}"

# 2a. the optimizers to run (OPTIMS, top of the cell) and their methods (benchmarks/methods.py : plain Python, no TPU)
import runpy
METHODS = runpy.run_path(f"{REPO}/benchmarks/methods.py")["METHODS"]
RUN = sorted({o.lower() for o in OPTIMS})
assert RUN and set(RUN) <= set(METHODS), f"STOP : OPTIMS = {OPTIMS}, the known optimizers are {sorted(METHODS)}"

# 2b. correctness : every check compares FORGE with PyTorch (benchmarks/tests.py), into results/<optim>/tests
for optim in RUN if RUN_CORRECTNESS_TESTS else ():
    print(f"\n========== correctness tests : {optim} ==========")
    OUT = f"{REPO}/results/{optim}/tests"
    code = run("benchmarks/tests.py", cwd = REPO, env = {**os.environ, "PYTHONPATH": f"{REPO}/src", "RESULTS": OUT, "OPTIM": optim})
    if code != 0 or not os.path.exists(f"{OUT}/tests_report.md"): print(f"tests.py ({optim}) crashed (see the error above)")
    else: print(f"\n{optim} : ALL CHECKS PASS" if "FAIL" not in open(f"{OUT}/tests_report.md").read() else f"\n{optim} : SOME CHECKS FAIL : see the list above and results/{optim}/tests/tests_report.md")

# 3. each optimizer and model : train + measure each method in its own process, then plots + table, into results/<optim>/<model>
for optim in RUN if RUN_BENCHMARKS else ():
    for prefix in ("mlp", "llama", "qwen"):
        print(f"\n========== {optim} : {prefix} ==========")
        OUT = f"{REPO}/results/{optim}/{prefix}"
        ENV = {**os.environ, "PYTHONPATH": f"{REPO}/src", "RESULTS": OUT, "MAX_SECONDS": "60", "OPTIM": optim}
        failed = [name for name in METHODS[optim] if run(f"benchmarks/{prefix}.py", name, cwd = REPO, env = ENV) != 0]
        if failed: print(f"\nFAILED : {optim} {prefix} {failed} (see the errors above). The report shows NA for them.")
        if run(f"benchmarks/{prefix}.py", "report", cwd = REPO, env = ENV) != 0: print(f"{optim} {prefix} : the report failed"); continue
        for png in ("headline", "loss", "memory"): display(Image(f"{OUT}/{prefix}_{png}.png"))
        display(Markdown(open(f"{OUT}/{prefix}_table.md").read()))

# 4. profile : where the time of one step goes (benchmarks/profiling.py), the same optimizers, models and methods, into the same results/<optim>/<model> folders
for optim in RUN if RUN_PROFILING else ():
    for prefix in ("mlp", "llama", "qwen"):
        print(f"\n========== {optim} : {prefix} : profile ==========")
        OUT = f"{REPO}/results/{optim}/{prefix}"
        ENV = {**os.environ, "PYTHONPATH": f"{REPO}/src:{REPO}/benchmarks", "RESULTS": OUT, "OPTIM": optim}
        failed = [name for name in METHODS[optim] if run("benchmarks/profiling.py", prefix, name, cwd = REPO, env = ENV) != 0]
        if failed: print(f"\nFAILED : {optim} {prefix} profile {failed} (see the errors above). The report shows NA for them.")
        if run("benchmarks/profiling.py", prefix, "report", cwd = REPO, env = ENV) != 0: print(f"{optim} {prefix} : the profile report failed"); continue
        for png in ("timeline", "host"): display(Image(f"{OUT}/profile_{prefix}_{png}.png"))
        display(Markdown(open(f"{OUT}/profile_{prefix}_table.md").read()))

# 5. download results.zip (folders <optim>/<model>/ : benchmarks and profiles, <optim>/tests/ : correctness). To select the folder : turn on 'Ask where to save each file' in your browser settings
os.makedirs(f"{REPO}/results", exist_ok = True)
files.download(shutil.make_archive("/content/results", "zip", f"{REPO}/results"))
