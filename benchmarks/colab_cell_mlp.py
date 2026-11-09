#@title FORGE-TPU · MLP benchmark on a Colab TPU v5e-1 (paste this whole file into ONE cell and run it)
# What it does, in order:
#   1. checks that this runtime has a TPU (without opening it, so the benchmark can use it later)
#   2. checks / installs the Python packages IN THE BACKGROUND while you pick the zip, so the TPU is not idle twice
#   3. asks for the GitHub zip (Code -> Download ZIP, of branch claude/great-heisenberg-xf572m), unpacks it
#   4. runs benchmarks/mlp.py: a correctness check against fp32 AdamW, then every method x every batch size,
#      each in its own process, printing every step
#   5. shows the tables and figures here, zips the results and downloads the zip (also if something fails half way)
# Nothing is downloaded: the MLP (like dk4248's examples/quickstart.py, 12 x 8192 by default) is built from scratch.

# ======================= knobs =======================
COMPILE_TORCH = True                       # True: torch.compile (FusedLinear itself stays eager, see mlp.py). False: eager (defer and fuse)
BATCH_SIZES   = [512, 2048, 8192]          # rows per batch. Predicted peaks <= ~8.3 GiB (fused / FORGE); vanilla adds its optimizer temporaries
MLP_WIDTH     = 8192                       # every layer is MLP_WIDTH x MLP_WIDTH (+ bias)
MLP_LAYERS    = 12                         # 12 x 8192: 805,404,672 parameters (weights 1.50 GiB)
WARMUP_STEPS  = 10                         # not timed (the first ones compile)
TIMED_STEPS   = 20                         # step time = median of these
METHODS       = ["vanilla AdamW", "fused AdamW", "FORGE (bf16)", "FORGE (int8)"]
# =====================================================

import glob, os, shutil, subprocess, sys, threading, time, zipfile
T0 = time.time()
def say(message): print(f"[{time.time() - T0:6.0f} s] {message}", flush=True)

# ---- 1. Is there a TPU? (device files only: importing torch or jax HERE would take the TPU away from the benchmark) ----
devices = sorted(glob.glob("/dev/accel*") + glob.glob("/dev/vfio/[0-9]*"))
if not devices:
    raise RuntimeError("No TPU in this runtime. Runtime -> Change runtime type -> TPU v5e-1, then run this cell again.")
say(f"TPU found: {devices}  (TPU_ACCELERATOR_TYPE={os.environ.get('TPU_ACCELERATOR_TYPE', 'not set')})")

# ---- 2. Packages, checked (and installed if missing) in the background ----
CHECK = r'''
import importlib.util, importlib.metadata as md
def version(p):
    if importlib.util.find_spec(p) is None: return "MISSING"
    try: return md.version(p)
    except Exception: return "present"
found = {p: version(p) for p in ("torch", "torch_tpu", "jax", "libtpu", "matplotlib")}
if found["torch"] != "MISSING":
    import torch   # does not open the TPU (TorchTPU defers that until a tensor goes to it); this process exits right after
    found["torch.tpu"] = "available" if hasattr(torch, "tpu") else "MISSING"
else:
    found["torch.tpu"] = "MISSING"
print(found)
'''
packages, setup_log = {}, []
def setup():
    def check():
        out = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True).stdout.strip().splitlines()
        return eval(out[-1]) if out else {"torch.tpu": "MISSING"}
    found = check()
    setup_log.append(f"packages: {found}")
    for package in ("matplotlib",):
        if found.get(package) == "MISSING":
            setup_log.append(f"installing {package} ...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", package])
    if found.get("torch.tpu") == "MISSING":
        # TorchTPU's wheel (Python 3.11+, Linux x86_64); tried only when PyTorch for TPU is not already in the runtime
        setup_log.append("PyTorch for TPU is missing: pip install torch_tpu ...")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch_tpu",
                            "--extra-index-url", "https://download.pytorch.org/whl/cpu"], capture_output=True, text=True)
        setup_log.append((r.stdout + r.stderr)[-1500:])
    if "MISSING" in (found.get("torch.tpu"), found.get("matplotlib")):
        found = check()
        setup_log.append(f"after installing: {found}")
    packages.update(found)
setup_thread = threading.Thread(target=setup, daemon=True)
setup_thread.start()
say("checking the Python packages in the background ...")

# ---- 3. The code: the GitHub zip ----
from google.colab import files
say("UPLOAD the zip now (GitHub -> branch claude/great-heisenberg-xf572m -> Code -> Download ZIP). The TPU waits for you.")
uploaded = files.upload()
if not uploaded:
    raise RuntimeError("No file uploaded.")
zip_name = next(iter(uploaded))
WORK = "/content/forge_run"
shutil.rmtree(WORK, ignore_errors=True)
with zipfile.ZipFile(zip_name) as z:
    z.extractall(WORK)
found = glob.glob(f"{WORK}/**/benchmarks/mlp.py", recursive=True)
if not found:
    raise RuntimeError(f"benchmarks/mlp.py is not in {zip_name}: is it the zip of branch claude/great-heisenberg-xf572m?")
REPO = os.path.dirname(os.path.dirname(found[0]))
say(f"code unpacked in {REPO}")

while setup_thread.is_alive():
    say("still checking / installing packages ...")
    setup_thread.join(timeout=15)
print("\n".join(setup_log), flush=True)
if packages.get("torch.tpu") != "available":
    raise RuntimeError("PyTorch for TPU (torch.tpu) is not available in this runtime and could not be installed; see the log above.")

# ---- 4. The benchmark (its own processes; this notebook never touches the TPU) ----
mode = "compiled" if COMPILE_TORCH else "eager"
OUT = f"/content/forge_mlp_results/{time.strftime('%Y%m%d-%H%M%S')}"
env = {**os.environ, "COMPILE_TORCH": "1" if COMPILE_TORCH else "0", "BATCH_SIZES": ",".join(map(str, BATCH_SIZES)),
       "MLP_WIDTH": str(MLP_WIDTH), "MLP_LAYERS": str(MLP_LAYERS),
       "WARMUP_STEPS": str(WARMUP_STEPS), "TIMED_STEPS": str(TIMED_STEPS), "METHODS": ";".join(METHODS),
       "OUT_DIR": OUT, "PYTHONUNBUFFERED": "1"}
say(f"starting: correctness check, then {len(METHODS)} methods x {len(BATCH_SIZES)} batch sizes, {mode}. Every step prints a line; "
    "a quiet stretch prints 'still working' every 30 s.")
child = None
try:
    child = subprocess.Popen([sys.executable, "-u", "benchmarks/mlp.py"], cwd=REPO, env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    for line in child.stdout:
        print(line, end="", flush=True)
    child.wait()
    say(f"benchmark finished (exit code {child.returncode})")
finally:
    if child and child.poll() is None:
        child.kill()  # the cell was stopped: stop the benchmark too, keep what is done
    # ---- 5. Show, zip, download (also after a failure or a stop: every finished run is already saved) ----
    if os.path.isdir(OUT):
        from IPython.display import Image, Markdown, display
        for check in sorted(glob.glob(f"{OUT}/*/log check.txt")):
            lines = [l for l in open(check) if l.startswith("[check]")]
            display(Markdown("### Correctness check\n\n```\n" + "".join(lines) + "```"))
        for summary in sorted(glob.glob(f"{OUT}/*/summary.md")):
            display(Markdown("### All batch sizes\n\n" + open(summary).read()))
        for picture in sorted(glob.glob(f"{OUT}/*/scaling.png")):
            display(Image(picture, width=1100))
        for table in sorted(glob.glob(f"{OUT}/*/batch_*/table.md"), key=lambda p: int(p.split("batch_")[-1].split("/")[0])):
            folder = os.path.dirname(table)
            display(Markdown(f"### {os.path.basename(folder)}\n\n" + open(table).read()))
            for picture in ("headline.png", "memory.png", "losses.png"):
                if os.path.exists(f"{folder}/{picture}"):
                    display(Image(f"{folder}/{picture}", width=1100))
        archive = shutil.make_archive(f"/content/forge_mlp_{mode}_{os.path.basename(OUT)}", "zip", OUT)
        say(f"results zipped: {archive} - downloading")
        files.download(archive)
    else:
        say("no results were written")
