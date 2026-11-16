# FORGE TPU : profile of MLP + Llama 1B + Qwen3 0.6B (where the time of one step goes), one Colab cell (TPU v5e-1 / v6e-1). ~12 min.
# Each model and method in its own process (benchmarks/profiling.py). Results : profile.zip with mlp/, llama/, qwen/ (profile_* files only)
import os, re, shutil, subprocess, sys, warnings, zipfile
from IPython.display import Image, Markdown, display
from google.colab import files
warnings.filterwarnings("ignore")

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
assert os.path.exists(f"{REPO}/src/forge_tpu/__init__.py") and os.path.exists(f"{REPO}/benchmarks/profiling.py"), f"STOP : the zip must hold src/ and benchmarks/ (at its top level or in one top folder), it holds {os.listdir(REPO)}"

# 3. profile : where the time of one step goes (benchmarks/profiling.py), the same models and methods, into the same results/<model> folders
for prefix in ("mlp", "llama", "qwen"):
    print(f"\n========== {prefix} : profile ==========")
    OUT = f"{REPO}/results/{prefix}"
    ENV = {**os.environ, "PYTHONPATH": f"{REPO}/src:{REPO}/benchmarks", "RESULTS": OUT}
    failed = [name for name in ("adamw", "adamw_fused", "forge_bf16", "forge_int8") if run("benchmarks/profiling.py", prefix, name, cwd = REPO, env = ENV) != 0]
    if failed: print(f"\nFAILED : {prefix} profile {failed} (see the errors above). The report shows the others.")
    if run("benchmarks/profiling.py", prefix, "report", cwd = REPO, env = ENV) != 0: print(f"{prefix} : the profile report failed"); continue
    for png in ("timeline", "host"): display(Image(f"{OUT}/profile_{prefix}_{png}.png"))
    display(Markdown(open(f"{OUT}/profile_{prefix}_table.md").read()))

# 4. download profile.zip (folders mlp/, llama/, qwen/). To select the folder : turn on 'Ask where to save each file' in your browser settings
files.download(shutil.make_archive("/content/profile", "zip", f"{REPO}/results"))
