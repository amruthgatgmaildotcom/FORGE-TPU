# FORGE TPU on Kaggle (TPU v5e-8, 8 chips) : the Llama-3.1-8B runs in ONE notebook cell. Paste this file into the cell. See benchmarks/benchmarks/kaggle/README.md for the Kaggle settings.
# Order : install -> TPU check -> 8-chip checks (smoke) -> correctness tests (1 chip) -> benchmarks -> profiles -> one-device memory -> results zip.
# Every run is a child process WITHOUT a time limit. A run that fails is written in the log and in logs/status.json, and the next run starts.
# Every result file is written to /kaggle/working/results/kaggle_v5e8 as soon as its run ends. The zip is made at the end in every case (also after an error here).
# This cell never asks for input and never raises. NOTE : Kaggle itself stops a TPU session after 9 h. Interactive session : set Session options -> Persistence -> "Files only" first, so /kaggle/working is kept after
# the session stops (Kaggle stops an idle session by itself after the cell ends). "Save & Run All (Commit)" also works : it keeps the output and ends by itself.
import datetime, glob, json, os, re, shutil, subprocess, sys, threading, time, traceback, zipfile

OPTIMS = ["adamw", "muon"]                                                                   # the optimizers to run, in this order (benchmarks/config.py has their methods)
PROFILE = ["adamw"]                                                                          # the optimizers that are also profiled
RUN_SMOKE, RUN_TESTS, RUN_BENCHMARKS, RUN_PROFILING = 1, 1, 1, 1                             # 1 : run that part, 0 : skip it
RUN_ENV = {"SEQ": "512", "SEED": "0", "MAX_STEPS": "200", "MAX_SECONDS": "1e9", "WARMUP": "10", "LR_WARMUP": "0"}   # benchmarks/config.py : 200 steps (a FIXED count : every chip runs the same steps), the first 10 not timed, no lr warmup
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("kaggle_working")
RESULTS, LOGS = f"{WORK}/results/kaggle_v5e8", f"{WORK}/logs"
REPO, VENV = "/tmp/TRAVELLER", "/tmp/forge_venv"
PY = f"{VENV}/bin/python"
PACKAGES = [["torch==2.13.0", "--index-url", "https://download.pytorch.org/whl/cpu"], ["torch_tpu==0.3.0", "jax==0.12.0.1", "transformers==5.23.0"]]
START, STATUS = time.time(), []
TEST_CPU = os.environ.get("KAGGLE_CELL_TEST_CPU") == "1"              # only to test this cell without a TPU (8 CPU processes, small models). Never set on Kaggle
if TEST_CPU: RUN_ENV["MAX_STEPS"] = os.environ.get("KAGGLE_STEPS", "12")                          # the CPU test : short
for old in (f"{WORK}/results", LOGS, f"{WORK}/forge_kaggle_results.zip"):        # a run before this one : moved to old_<time>/, so its files never mix with this run
    if os.path.exists(old): os.makedirs(f"{WORK}/old_{int(time.time())}", exist_ok = True); shutil.move(old, f"{WORK}/old_{int(time.time())}/")
os.makedirs(RESULTS, exist_ok = True); os.makedirs(LOGS, exist_ok = True)
LOG = open(f"{LOGS}/cell_log.txt", "a", buffering = 1)
NOISE = re.compile(r"^\[?[IWE]\d{4} |^WARNING: All log messages|^Successfully renamed PrivateUse1|^Registered Python module|^Device type:|^Initializing TPU|socket\.cpp|OMP_NUM_THREADS|^\*{5}|tcmalloc|_tcmalloc_check|^USDT:")
QUIET = {"PYTHONWARNINGS": "ignore", "TF_CPP_MIN_LOG_LEVEL": "3", "TPU_STDERR_LOG_LEVEL": "3", "TPU_MIN_LOG_LEVEL": "3", "GLOG_minloglevel": "3", "GRPC_VERBOSITY": "ERROR"}



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush = True); LOG.write(line + "\n")

def elapsed(): return time.time() - START
def clock(): return str(datetime.timedelta(seconds = int(elapsed())))

def run(cmd, kind, name, env = None, cwd = None):
    # a child process : its output is shown (without the TPU runtime noise) and saved in logs/<name>.txt (all of it). No time limit : it runs until it ends
    path = f"{LOGS}/{name}.txt"
    log(f"\n[{clock()}] ===== {name} =====")
    t0 = time.time()
    try:
        p = subprocess.Popen(cmd, cwd = cwd or (REPO if os.path.isdir(REPO) else None), env = env, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True, errors = "replace", start_new_session = True)
    except Exception as e:
        log(f"cannot start {name} : {e}"); STATUS.append(dict(run = name, code = "not started", minutes = 0)); return -1
    def reader():
        with open(path, "w") as f:
            for line in p.stdout:
                f.write(line)
                if not NOISE.search(line): print(line, end = "", flush = True); LOG.write(line)
    t = threading.Thread(target = reader, daemon = True); t.start()
    code = p.wait()
    t.join()
    minutes = round((time.time() - t0) / 60, 1)
    STATUS.append(dict(run = name, code = code, minutes = minutes))
    json.dump(STATUS, open(f"{LOGS}/status.json", "w"), indent = 1)
    log(f"[{clock()}] {name} : {'OK' if code == 0 else 'FAILED (' + str(code) + ')'} in {minutes} min")
    return code

def cleanup():
    # nothing of a finished run may keep the TPU : kill what is left of it, remove libtpu's lock file
    for pattern in ("kaggle/run.py", "torch.distributed.run", "benchmarks/tests.py"):
        try: subprocess.run(["pkill", "-9", "-f", pattern], stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
        except Exception: pass
    time.sleep(5)
    for lock in glob.glob("/tmp/libtpu_lockfile*"):
        try: os.remove(lock)
        except Exception: pass

def tpu_present():
    # the TPU, before anything is installed : its device files (/dev/accel*, /dev/vfio/<n>) or the TPU variables of the VM
    devices = sorted(glob.glob("/dev/accel*") + glob.glob("/dev/vfio/[0-9]*"))
    names = {k: v for k, v in os.environ.items() if k in ("TPU_ACCELERATOR_TYPE", "TPU_NAME", "TPU_WORKER_ID", "TPU_SKIP_MDS_QUERY")}
    log(f"TPU check : device files {devices or 'none'}, variables {names or 'none'}")
    return bool(devices or names)

def show_failure(name):
    # the end of the logs of chips 1..7 (chip 0 is in the main output)
    for f in sorted(glob.glob(f"{LOGS}/{name}/rank*.log")):
        tail = open(f, errors = "replace").read().strip().splitlines()[-15:]
        if tail: log(f"--- {os.path.basename(f)} (last lines) ---\n" + "\n".join(tail))



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def find_repo():
    # the repo zip added as input : Kaggle unpacks it under /kaggle/input. If it is still a zip, unpack it here
    def walk(top):
        for root, dirs, files in os.walk(top):
            if os.path.isfile(f"{root}/benchmarks/kaggle/run.py") and os.path.isfile(f"{root}/src/forge_tpu/__init__.py"): return root
    found = walk("/kaggle/input")
    if not found:
        for z in glob.glob("/kaggle/input/**/*.zip", recursive = True):
            try: zipfile.ZipFile(z).extractall("/tmp/repo_zip")
            except Exception as e: log(f"cannot unpack {z} : {e}")
        found = walk("/tmp/repo_zip")
    if not found: return False
    shutil.rmtree(REPO, ignore_errors = True); shutil.copytree(found, REPO)
    log(f"repo : {found} -> {REPO}")
    return True

def install():
    # a clean Python (uv) with exactly the versions FORGE was checked with : the Kaggle image's own torch / jax / torch_xla are not used
    uv = [sys.executable, "-m", "uv"]
    if run([sys.executable, "-m", "pip", "install", "-q", "uv"], "install", "install_uv") != 0: return False
    if not os.path.exists(PY) and run(uv + ["venv", "--python", "3.13", VENV], "install", "install_python") != 0:   # a second run of the cell keeps the venv
        if sys.version_info < (3, 11) or run([sys.executable, "-m", "venv", VENV], "install", "install_python_venv") != 0: return False
    for i, packages in enumerate(PACKAGES):
        if run(uv + ["pip", "install", "--python", PY, *packages], "install", f"install_packages_{i + 1}") != 0: return False
    return True

def child_env():
    env = dict(os.environ)
    removed = [k for k in env if k in ("PJRT_DEVICE", "TPU_VISIBLE_CHIPS", "TPU_VISIBLE_DEVICES", "TPU_PROCESS_BOUNDS", "TPU_CHIPS_PER_PROCESS_BOUNDS", "TPU_PROCESS_ADDRESSES",
                                       "TPU_PROCESS_PORT", "XLA_FLAGS", "LIBTPU_INIT_ARGS", "PYTHONPATH", "JAX_PLATFORMS", "LD_PRELOAD", "PYTHONHOME", "VIRTUAL_ENV") or k.startswith(("TORCH_TPU_", "XRT_"))]
    for k in removed: env.pop(k)
    if removed: log(f"environment variables not passed to the runs : {removed}")
    env.update(QUIET, PATH = f"{VENV}/bin:" + env.get("PATH", ""), PYTHONPATH = f"{REPO}/src:{REPO}/benchmarks:{REPO}/benchmarks/kaggle", PYTHONUNBUFFERED = "1",
               JAX_PLATFORMS = "cpu")                # JAX only traces the Pallas kernel (torch_tpu runs it) : it must never take the TPU itself
    tcmalloc = sorted(glob.glob("/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4*") + glob.glob("/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4*"))
    if not tcmalloc:
        try: subprocess.run(["apt-get", "install", "-y", "-q", "libtcmalloc-minimal4"], stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL, stdin = subprocess.DEVNULL)
        except Exception as e: log(f"apt-get install libtcmalloc-minimal4 : {type(e).__name__}")
        tcmalloc = sorted(glob.glob("/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4*"))
    if tcmalloc: env["LD_PRELOAD"] = tcmalloc[0]                        # torch_tpu asks for it (host speed), as on Colab
    log(f"tcmalloc : {tcmalloc[0] if tcmalloc else 'not found (torch_tpu warns, the runs still work)'}")
    return env

CHECK = r'''
import importlib.metadata as md, os, torch
print("packages :", {p: md.version(p) for p in ("torch", "torch_tpu", "jax", "jaxlib", "libtpu", "transformers")})
dev = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else None
assert dev is not None and dev.type == "tpu", f"torch sees no TPU (accelerator = {dev})"
print("TPU devices :", torch.accelerator.device_count(), "| TPU_ACCELERATOR_TYPE :", os.environ.get("TPU_ACCELERATOR_TYPE", "?"))
x = torch.ones(1024, 1024, device = "tpu", dtype = torch.bfloat16)
assert (x @ x)[0, 0].item() == 1024, "TPU matmul gives a wrong value"
print("TPU OK")
'''

LAUNCH = r'''
import importlib, json, os, torch
try: from torch.tpu import distributed as d
except Exception: d = importlib.import_module("torch_tpu._internal.distributed.distributed")
d.set_tpu_launch_env(nproc_per_node = 8)
print("LAUNCH_ENV " + json.dumps({k: v for k, v in os.environ.items() if k.startswith("TORCH_TPU_")}))
'''

def launch_env(env):
    # torch_tpu's launch variables (topology, one address per chip), made fresh for every 8-chip run, in a short process that then frees the TPU
    if TEST_CPU: return {}
    try: out = subprocess.run([PY, "-c", LAUNCH], env = env, cwd = REPO, capture_output = True, text = True)
    except Exception as e: log(f"torch_tpu launch variables failed : {type(e).__name__}"); return None
    line = next((l for l in out.stdout.splitlines() if l.startswith("LAUNCH_ENV ")), None)
    if not line: log("torch_tpu launch variables failed :\n" + (out.stdout + out.stderr)[-3000:]); return None
    return json.loads(line[len("LAUNCH_ENV "):])

TP = {"failed_in_a_row": 0, "broken": False}

def tp_run(env, name, kind, args, extra):
    # one 8-chip run (torchrun : one process per chip). After 3 failures in a row, no more 8-chip runs (something is broken : do not waste the session)
    if TP["broken"]: log(f"skipped {name} : the 8-chip runs failed 3 times in a row"); STATUS.append(dict(run = name, code = "skipped")); return "skipped"
    cleanup()
    launch = launch_env(env)
    if launch is None: code = "no launch env"
    else:
        cmd = [PY, "-m", "torch.distributed.run", "--standalone", "--nnodes", "1", "--nproc_per_node", "8", "--max_restarts", "0", f"{REPO}/benchmarks/kaggle/run.py", kind, *args]
        code = run(cmd, kind, name, env = {**env, **launch, **extra, "RUN_LOGS": f"{LOGS}/{name}"})
    cleanup()
    if code != 0: show_failure(name); TP["failed_in_a_row"] += 1
    else: TP["failed_in_a_row"] = 0
    if TP["failed_in_a_row"] >= 3: TP["broken"] = True
    return code



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def main():
    log(f"[{clock()}] FORGE TPU on Kaggle : start {datetime.datetime.now()}, results in {RESULTS}")
    if not TEST_CPU and not tpu_present(): log("STOP : this session has no TPU (Session options -> Accelerator -> TPU v5e-8)"); return
    if not find_repo(): log("STOP : no repo under /kaggle/input (add the GitHub zip as input, see benchmarks/kaggle/README.md)"); return
    if not install(): log("STOP : the install failed (is Internet on in the notebook settings ?)"); return
    env = child_env()
    if TEST_CPU: env.update(TP_DEVICE = "cpu", KAGGLE_TEST_TINY = "1")
    elif run([PY, "-c", CHECK], "check", "tpu_check", env = env) != 0: log("STOP : no working TPU (Accelerator must be TPU v5e-8)"); return
    cleanup()
    methods = json.loads(subprocess.run([PY, "-c", "import json, config; print(json.dumps(config.METHODS))"], env = env, cwd = f"{REPO}/benchmarks", capture_output = True, text = True).stdout.strip().splitlines()[-1])
    optims = [o for o in OPTIMS if o in methods]
    if TEST_CPU: methods = {o: [m[0], m[-1]] for o, m in methods.items()}                  # the CPU test : the first and the last method of each optimizer
    run_env = lambda optim: {**RUN_ENV, "OPTIM": optim, "RESULTS": RESULTS}

    if RUN_SMOKE:
        for optim in optims: tp_run(env, f"smoke_{optim}", "smoke", [], run_env(optim))
    def tests(optim):
        if not RUN_TESTS: return
        cleanup(); run([PY, "benchmarks/tests.py"], "tests", f"tests_{optim}", env = {**env, **run_env(optim)}); cleanup()
    def stage(kind, optims):
        for optim in optims:
            if kind == "bench": tests(optim)                                                 # correctness of this optimizer first (1 chip)
            for method in methods[optim]: tp_run(env, f"{kind}_{optim}_llama8b_{method}", kind, ["llama8b", method], run_env(optim))
            make_zip()
    if RUN_BENCHMARKS: stage("bench", optims)
    elif RUN_TESTS:
        for optim in optims: tests(optim)
    if RUN_PROFILING: stage("profile", [o for o in optims if o in PROFILE])

def make_zip():
    # results/ + logs/ in one zip, made again after every model (so a zip of everything so far always exists)
    try:
        with zipfile.ZipFile(f"{WORK}/forge_kaggle_results.zip.tmp", "w", zipfile.ZIP_DEFLATED) as z:
            for top in (f"{WORK}/results", LOGS):
                for root, dirs, files in os.walk(top):
                    for f in files: z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), WORK))
        os.replace(f"{WORK}/forge_kaggle_results.zip.tmp", f"{WORK}/forge_kaggle_results.zip")
    except Exception: log("zip failed :\n" + traceback.format_exc())

def finish():
    try:
        if os.path.exists(PY) and os.path.isdir(REPO):
            e = child_env()
            run([PY, "benchmarks/kaggle/one_device.py", RESULTS], "report", "one_device", env = e)   # the memory of one device : <run>_1dev.json
    except Exception: log(traceback.format_exc())
    log(f"\n[{clock()}] ===== summary =====")
    for s in STATUS: log(f"{s['run']:45s} {str(s['code']):12s} {s.get('minutes', '')} min")
    json.dump(STATUS, open(f"{LOGS}/status.json", "w"), indent = 1)
    LOG.flush()
    make_zip()
    log(f"[{clock()}] done : {WORK}/forge_kaggle_results.zip (results/kaggle_v5e8/ and logs/). The session ends now.")

try: main()
except BaseException: log("ERROR in the cell (the results so far are kept) :\n" + traceback.format_exc())
finally:
    try: cleanup()
    except Exception: pass
    try: finish()
    except BaseException: print(traceback.format_exc())
