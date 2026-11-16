import os

# The optimizers of the benchmarks and their methods. Plain Python (no torch, no TPU) : the Colab cells read this file too, to know what to run.
# One run of a benchmark script uses ONE optimizer : the environment variable OPTIM (default adamw). Its results go to results/<optim>/<model>.
# To add an optimizer : its methods here (the first one is the reference of the loss table : vanilla torch), its labels, then its torch optimizer in mlp.torch_optimizer.
METHODS = {"adamw": ("adamw", "adamw_fused", "forge_bf16", "forge_int8"),
           "sgd":   ("sgd", "sgd_fused", "forge_sgd")}
LABELS = {"adamw": "vanilla AdamW", "adamw_fused": "fused AdamW", "forge_bf16": "FORGE (bf16)", "forge_int8": "FORGE (int8)",
          "sgd": "vanilla SGD", "sgd_fused": "fused SGD", "forge_sgd": "FORGE (SGD)"}

def optim():
    # the optimizer of this run, from OPTIM ("adamw", "sgd", any case)
    name = os.environ.get("OPTIM", "adamw").lower()
    if name not in METHODS: raise SystemExit(f"OPTIM={name!r} : unknown optimizer, use one of {sorted(METHODS)}")
    return name

OPTIM = optim()
OPTIMIZERS = METHODS[OPTIM]                                            # the methods of this run, the reference (vanilla torch) first
