import os

#   Readme.md
#     The one place that says WHAT a benchmark runs and WHERE its result goes. Plain Python (no torch, no TPU) : the Colab cell and report.py read it too.
#     One process runs ONE method of ONE optimizer on ONE model (so the memory of a run is never mixed with another one). What it runs comes from the environment :
#        OPTIM        adamw | sgd | muon                 (default adamw)
#        SEED         the seed of the model init and of the data, the same for every method of a run      (default 0)
#        SEQ          tokens per sequence of Llama and Qwen, batch 1 (the MLP always has 4096 tokens per step)   (default 512)
#        MAX_STEPS, MAX_SECONDS   a training run stops after MAX_STEPS steps or after MAX_SECONDS seconds, whatever comes first   (default 300, 100)
#        WARMUP       the first WARMUP steps are not timed   (default 20)
#        LR_WARMUP    the lr goes up linearly over the first LR_WARMUP steps, the same for every method; 0 : no lr warmup   (default WARMUP)
#        RESULTS      the results folder   (default results/ of the repo; benchmarks/kaggle/ gives results/kaggle_v5e8)
#     Every file the benchmarks write (and nothing else) :
#        results/<optim>/<model>/<method>_seed<N>.json     train.py       one training run : the loss, time and memory of every step, and the summary numbers
#        results/<optim>/<model>/profile_<method>.json     profiling.py   where the time of one step goes
#        results/<optim>/tests.md                          tests.py       FORGE against PyTorch, every check
#        results/tiler/<model>.json, results/tiler/tiles.md   tiler.py    the fastest kernel tiles of every layer shape
#        results/summary.md, assets/*                      report.py      the tables and the figures, made from the files above only
#     Llama-3.1-8B on a TPU v5e-8 (8 chips, tensor parallel, benchmarks/kaggle/) writes the same files into results/kaggle_v5e8/, and two more :
#        results/kaggle_v5e8/<optim>/smoke.md                              kaggle/run.py smoke    the 8-chip checks
#        results/kaggle_v5e8/<optim>/llama8b/<method>_seed<N>_1dev.json    kaggle/one_device.py   the same run, memory CALCULATED for one device

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS, ASSETS = os.environ.get("RESULTS", os.path.join(ROOT, "results")), os.path.join(ROOT, "assets")



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



METHODS = {"adamw": ("adamw", "adamw_fused", "forge_bf16", "forge_int8"),                 # the first method of each optimizer is vanilla torch : the reference of every comparison
           "sgd":   ("sgd", "sgd_fused", "forge_sgd"),
           "muon":  ("muon", "forge_muon")}                                                  # Muon on the 2D weights of the layers, fused AdamW on the rest (src/forge_tpu/muon.py)
BEST = {"adamw": "adamw_fused", "sgd": "sgd_fused", "muon": "muon"}                          # the strongest torch method of each optimizer
LABELS = {"adamw": "vanilla AdamW", "adamw_fused": "fused AdamW", "forge_bf16": "FORGE (bf16)", "forge_int8": "FORGE (int8)",
          "sgd": "vanilla SGD", "sgd_fused": "fused SGD", "forge_sgd": "FORGE (SGD)",
          "muon": "Muon", "forge_muon": "FORGE (Muon)"}
MODELS = {"mlp": "MLP-1.1B", "llama": "Llama-3.2-1B", "qwen": "Qwen3-0.6B"}                  # benchmarks/<model>.py




#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



OPTIM = os.environ.get("OPTIM", "adamw").lower()
if OPTIM not in METHODS: raise SystemExit(f"OPTIM={OPTIM!r} : unknown optimizer, use one of {sorted(METHODS)}")
SEED = int(os.environ.get("SEED", 0))
SEQ = int(os.environ.get("SEQ", 512))
MAX_STEPS = int(os.environ.get("MAX_STEPS", 300))
MAX_SECONDS = float(os.environ.get("MAX_SECONDS", 100))
WARMUP = int(os.environ.get("WARMUP", 20))
LR_WARMUP = int(os.environ.get("LR_WARMUP", WARMUP))

def run_path(optim, model, method, seed): return os.path.join(RESULTS, optim, model, f"{method}_seed{seed}.json")
def profile_path(optim, model, method):   return os.path.join(RESULTS, optim, model, f"profile_{method}.json")
def tests_path(optim):                    return os.path.join(RESULTS, optim, "tests.md")
def tiler_path(name):                     return os.path.join(RESULTS, "tiler", name)
