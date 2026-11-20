# FORGE TPU on Kaggle : Llama-3.1-8B on a TPU v5e-8 (8 chips)

Llama-3.1-8B is too large for one 16 GiB TPU v5e chip. These files split it over the 8 chips of a Kaggle TPU v5e-8 (tensor parallel). `src/` is not changed : each chip holds normal `nn.Linear` / `FusedLinear` layers with 1/8 of each weight, and FORGE updates each shard on its own chip. The measurement is the same as on one chip (`benchmarks/train.py`, `benchmarks/profiling.py`, `benchmarks/tests.py`).

| File | What it does |
|---|---|
| `kaggle_cell.py` | The cell : install, TPU check, 8-chip checks, correctness tests, benchmarks, profiles, one-device memory, zip |
| `tp.py` | The tensor-parallel layers (collectives with autograd), the sharded Llama, Muon's Newton–Schulz over the shards (`apply_muon`) |
| `llama8b.py` | The model : Llama-3.1-8B, the data and the settings of `benchmarks/llama.py` |
| `run.py` | One 8-chip run (torchrun, one process per chip) : `smoke`, `bench llama8b <method>`, `profile llama8b <method>` |
| `one_device.py` | The memory of one device, calculated from the memory of one chip : `<run>_1dev.json` |

## Steps

1. GitHub : **Code -> Download ZIP** of this repository.
2. Kaggle : **Create -> New Dataset** -> upload the zip (do not unzip it) -> any title -> **Create**. Kaggle unpacks it.
3. Kaggle : **Create -> New Notebook** -> **Add Input** -> the dataset of step 2.
4. Right panel, **Session options** : **Accelerator = TPU v5e-8**, **Internet = On**, **Persistence = Files only** (keeps `/kaggle/working` after the session stops).
5. Paste all of `benchmarks/kaggle/kaggle_cell.py` into one cell. Delete the other cells. When the TPU session connects, run the cell.
6. After the cell ends, Kaggle stops the idle session. Next time you open the notebook : right panel **Output -> /kaggle/working -> forge_kaggle_results.zip** -> download. Unpack it into the repository : it holds `results/kaggle_v5e8/` (and `logs/`).

## What it runs

`OPTIMS = ["adamw", "muon"]`, `PROFILE = ["adamw"]`. Every run : batch 1, sequence 512, seed 0, 200 steps (the first 10 not timed), no lr warmup (`RUN_ENV` at the top of the cell).

1. Install : a clean Python 3.13 (uv) with torch 2.13.0 (CPU build), torch_tpu 0.3.0, jax 0.12.0.1, transformers 5.23.0.
2. TPU check (1 process).
3. The 8-chip checks, for each optimizer : the collectives, the memory of each chip, and each method on a small Llama in tensor parallel against the same model on 1 chip (gradients and updates) -> `results/kaggle_v5e8/<optim>/smoke.md`.
4. For each optimizer : `benchmarks/tests.py` (1 chip) -> `<optim>/tests.md`, then each method -> `<optim>/llama8b/<method>_seed0.json`. Each result has the peak of each chip : `peak_per_chip_gib` (the timed steps) and `peak_synced_per_chip_gib` (one step with a sync after each layer : no memory of ops that the host issued ahead).
5. The profiles of AdamW -> `adamw/llama8b/profile_<method>.json`.
6. `one_device.py` -> `<optim>/llama8b/<method>_seed0_1dev.json`, then the zip. `logs/` has all the output and `status.json` (each run : OK or FAILED).

A run that fails goes into `status.json`, and the next run starts. After 3 failed 8-chip runs in a row, no more 8-chip runs start.

Muon with tensor parallel : Newton–Schulz needs the full matrix. For each group of 8 matrices, each chip gathers their full momenta, chip j orthogonalizes matrix j, and a sum over the chips gives each chip the result (`tp.apply_muon`, the same code for both Muon methods).

To test this code without a TPU (8 CPU processes, a small model) : `TP_DEVICE=cpu KAGGLE_TEST_TINY=1 python -m torch.distributed.run --standalone --nproc_per_node 8 benchmarks/kaggle/run.py bench llama8b forge_bf16`, with `PYTHONPATH=src:benchmarks:benchmarks/kaggle`.
