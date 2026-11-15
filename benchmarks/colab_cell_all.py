# FORGE TPU : MLP + Llama 1B + Qwen3 0.6B benchmarks, then the profile of the same runs (correctness tests : RUN_CORRECTNESS_TESTS = 1), one Colab cell (TPU v5e-1 / v6e-1). ~30 min (18 benchmarks + 12 profile).
# Each run stops after MAX_SECONDS (60 here, 180 in the single-model cells), so the 12 runs fit in the free tier. Results : results.zip with mlp/, llama/, qwen/
import os, re, shutil, subprocess, sys, warnings, zipfile
from IPython.display import Image, Markdown, display
from google.colab import files
warnings.filterwarnings("ignore")

RUN_CORRECTNESS_TESTS, RUN_BENCHMARKS, RUN_PROFILING = 0, 1, 1        # 1 : run that part, 0 : skip it. Tests : benchmarks/tests.py, a few minutes, into results/tests

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

# 2. the repo zip (benchmarks/, src/, README.md at its top level) -> /content/TRAVELLER
REPO = "/content/TRAVELLER"
shutil.rmtree(REPO, ignore_errors = True)
zipfile.ZipFile(os.path.abspath(next(iter(files.upload())))).extractall(REPO)
inner = os.listdir(REPO)
if len(inner) == 1 and os.path.isdir(f"{REPO}/{inner[0]}"):          # a GitHub zip (Code > Download ZIP) has one top folder : move its content up
    top = f"{REPO}/{inner[0]}"
    for name in os.listdir(top): shutil.move(f"{top}/{name}", f"{REPO}/{name}")
    os.rmdir(top)
assert os.path.exists(f"{REPO}/src/forge_tpu/__init__.py") and os.path.exists(f"{REPO}/benchmarks/qwen.py"), f"STOP : the zip must hold src/ and benchmarks/ at its top level, it holds {os.listdir(REPO)}"

# 2b. correctness : every check compares FORGE with PyTorch (benchmarks/tests.py), into results/tests
if RUN_CORRECTNESS_TESTS:
    print("\n========== correctness tests ==========")
    OUT = f"{REPO}/results/tests"
    code = run("benchmarks/tests.py", cwd = REPO, env = {**os.environ, "PYTHONPATH": f"{REPO}/src", "RESULTS": OUT})
    if code != 0 or not os.path.exists(f"{OUT}/tests_report.md"): print("tests.py crashed (see the error above)")
    else: print("\nALL CHECKS PASS" if "FAIL" not in open(f"{OUT}/tests_report.md").read() else "\nSOME CHECKS FAIL : see the list above and results/tests/tests_report.md")

# 3. each model : train + measure each optimizer in its own process, then plots + table, into results/<model>
for prefix in ("mlp", "llama", "qwen") if RUN_BENCHMARKS else ():
    print(f"\n========== {prefix} ==========")
    OUT = f"{REPO}/results/{prefix}"
    ENV = {**os.environ, "PYTHONPATH": f"{REPO}/src", "RESULTS": OUT, "MAX_SECONDS": "60"}
    failed = [name for name in ("adamw", "adamw_fused", "forge_bf16", "forge_int8") if run(f"benchmarks/{prefix}.py", name, cwd = REPO, env = ENV) != 0]
    if failed: print(f"\nFAILED : {prefix} {failed} (see the errors above). The report shows the others.")
    if run(f"benchmarks/{prefix}.py", "report", cwd = REPO, env = ENV) != 0: print(f"{prefix} : the report failed"); continue
    for png in ("headline", "loss", "memory"): display(Image(f"{OUT}/{prefix}_{png}.png"))
    display(Markdown(open(f"{OUT}/{prefix}_table.md").read()))

# 4. profile : where the time of one step goes (benchmarks/profiling.py), the same models and methods, into the same results/<model> folders
for prefix in ("mlp", "llama", "qwen") if RUN_PROFILING else ():
    print(f"\n========== {prefix} : profile ==========")
    OUT = f"{REPO}/results/{prefix}"
    ENV = {**os.environ, "PYTHONPATH": f"{REPO}/src:{REPO}/benchmarks", "RESULTS": OUT}
    failed = [name for name in ("adamw", "adamw_fused", "forge_bf16", "forge_int8") if run("benchmarks/profiling.py", prefix, name, cwd = REPO, env = ENV) != 0]
    if failed: print(f"\nFAILED : {prefix} profile {failed} (see the errors above). The report shows the others.")
    if run("benchmarks/profiling.py", prefix, "report", cwd = REPO, env = ENV) != 0: print(f"{prefix} : the profile report failed"); continue
    for png in ("timeline", "host"): display(Image(f"{OUT}/profile_{prefix}_{png}.png"))
    display(Markdown(open(f"{OUT}/profile_{prefix}_table.md").read()))

# 5. download results.zip (folders mlp/, llama/, qwen/ : benchmarks and profiles, tests/ : correctness). To select the folder : turn on 'Ask where to save each file' in your browser settings
os.makedirs(f"{REPO}/results", exist_ok = True)
files.download(shutil.make_archive("/content/results", "zip", f"{REPO}/results"))
