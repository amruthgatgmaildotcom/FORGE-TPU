# FORGE TPU : correctness tests (FORGE against PyTorch), one Colab cell (TPU v5e-1 / v6e-1). A few minutes.
import os, re, shutil, subprocess, sys, warnings, zipfile
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

# 2. the repo zip (benchmarks/, src/, README.md at its top level) -> /content/TRAVELLER
REPO = "/content/TRAVELLER"
shutil.rmtree(REPO, ignore_errors = True)
zipfile.ZipFile(os.path.abspath(next(iter(files.upload())))).extractall(REPO)
assert os.path.exists(f"{REPO}/src/forge_tpu/__init__.py") and os.path.exists(f"{REPO}/benchmarks/tests.py"), f"STOP : the zip must hold src/ and benchmarks/ at its top level, it holds {os.listdir(REPO)}"
OUT = f"{REPO}/results/tests"
ENV = {**os.environ, "PYTHONPATH": f"{REPO}/src", "RESULTS": OUT}

# 3. run every check (output : PASS / FAIL per case, a model table, the list of failures)
code = run("benchmarks/tests.py", cwd = REPO, env = ENV)
assert code == 0 and os.path.exists(f"{OUT}/tests_report.md"), "STOP : tests.py crashed (see the error above)"
print("\nALL CHECKS PASS" if "FAIL" not in open(f"{OUT}/tests_report.md").read() else "\nSOME CHECKS FAIL : see the list above and tests_report.md")

# 4. download the report (to select the folder : turn on 'Ask where to save each file' in your browser settings)
files.download(shutil.make_archive("/content/forge_tests_results", "zip", OUT))
