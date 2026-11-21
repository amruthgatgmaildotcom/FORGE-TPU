<div align="center">

# 🔥 FORGE · TPU

### Fused On-Register Gradient Elimination on Google TPU v5e

*The weight gradient does not go to HBM. The optimizer step has almost no work.*

[![TPU v5e](https://img.shields.io/badge/Google-TPU_v5e-4285F4.svg?style=for-the-badge&logo=google&logoColor=white)](https://cloud.google.com/tpu/docs/v5e)
[![JAX Pallas](https://img.shields.io/badge/JAX-Pallas-7b3fe4.svg?style=for-the-badge)](https://docs.jax.dev/en/latest/pallas/index.html)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13_·_torch__tpu-ee4c2c.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

<p align="center"><b><a href="#-highlights">Highlights</a> · <a href="#%EF%B8%8F-the-main-limit--16-gib-of-hbm">HBM limit</a> · <a href="#-llama-31-8b-tpu-v5e-8">Llama-3.1-8B</a> · <a href="#-key-results-llama-32-1b-one-tpu-v5e-chip">Llama-3.2-1B</a> · <a href="#-every-model">Every model</a> · <a href="#%EF%B8%8F-where-the-time-goes">Time</a> · <a href="#-muon">Muon</a> · <a href="#-how-it-works-on-tpu">How it works</a></b></p>

</div>

---

FORGE puts the optimizer step into the backward pass. One Pallas kernel calculates the weight gradient one tile at a time, in VMEM. The same kernel updates `W`, `m` and `v` of the tile and writes them back into the same HBM buffers. **The full weight gradient is never in HBM.**

## ✨ Highlights

Each number below is against vanilla PyTorch AdamW, then against fused AdamW, on the same TPU.

- **A new FORGE for TPU, written from the start in Pallas, on a new software stack (`torch_tpu` 0.3).**
  - **In place : `W` and the moments go to the kernel as donated HBM buffers. A check on the first launch warns if XLA copies them.**
  - **Fast launch : the first launch of each layer shape is caught inside `torch_tpu` and replayed. The replay skips about 160 µs of checks for each layer; the launch itself takes about 21 µs (measured during development).**
- **Llama-3.1-8B, one device :** the peak memory decreases by **{{8b.forge_bf16.dmem:.0f}}%** ({{8b.adamw.peak:.1f}} → {{8b.forge_bf16.peak:.1f}} GiB) and by **{{8b.forge_int8.dmem:.0f}}%** with int8 moments. Against fused AdamW : −{{8b.forge_bf16.dmem_best:.0f}}%, −{{8b.forge_int8.dmem_best:.0f}}% with int8 moments. [More details](#-llama-31-8b-tpu-v5e-8).
- **Less memory :** FORGE decreases the peak memory on each model.
  - Llama-3.2-1B : **−{{llama.forge_bf16.dmem:.0f}}%** ({{llama.adamw.peak:.2f}} → {{llama.forge_bf16.peak:.2f}} GiB), **−{{llama.forge_int8.dmem:.0f}}%** with int8 moments. Against fused AdamW : −{{llama.forge_bf16.dmem_best:.0f}}%, −{{llama.forge_int8.dmem_best:.0f}}%.
  - MLP-1.1B : **−{{mlp.forge_bf16.dmem:.0f}}%** ({{mlp.adamw.peak:.2f}} → {{mlp.forge_bf16.peak:.2f}} GiB), **−{{mlp.forge_int8.dmem:.0f}}%** with int8 moments. Against fused AdamW : −{{mlp.forge_bf16.dmem_best:.0f}}%, −{{mlp.forge_int8.dmem_best:.0f}}%.
  - Muon, against standard Muon : **−{{muon.llama.forge_muon.dmem:.0f}}%** on Llama-3.2-1B, **−{{8b_muon.forge_muon.dmem:.0f}}%** on Llama-3.1-8B (one device, calculated).
- **Less time :** FORGE decreases the step time on each model of one chip. FORGE (bf16) is also faster than fused AdamW, except on Qwen3-0.6B ([see why](#-every-model)).
  - Llama-3.2-1B : **−{{llama.forge_bf16.dstep:.0f}}%** ({{llama.adamw.step_ms:.1f}} → {{llama.forge_bf16.step_ms:.1f}} ms). Against fused AdamW : −{{llama.forge_bf16.dstep_best:.0f}}%.
  - MLP-1.1B : **−{{mlp.forge_bf16.dstep:.0f}}%** ({{mlp.adamw.step_ms:.1f}} → {{mlp.forge_bf16.step_ms:.1f}} ms). Against fused AdamW : −{{mlp.forge_bf16.dstep_best:.0f}}%. The chip runs at **{{mlp.forge_bf16.mfu:.0f}}%** of its bf16 peak (model FLOPs).
  - Optimizer step, Llama-3.2-1B : **−{{llama.opt_cut:.0f}}%** ({{prof.llama.adamw.opt:.1f}} → {{prof.llama.forge_bf16.opt:.1f}} ms). Against fused AdamW : −{{llama.opt_cut_best:.0f}}%.
  - Muon, against standard Muon : **−{{muon.llama.forge_muon.dstep:.0f}}%** on Llama-3.2-1B.
- **The main limit is the HBM.** One TPU v5e chip has only 16 GiB of HBM. [See below](#%EF%B8%8F-the-main-limit--16-gib-of-hbm).

## ⚠️ The main limit : 16 GiB of HBM

The largest disadvantage of this work is the small HBM of one TPU v5e chip (16 GiB). In these results, FORGE gives the most improvement on large layers and on large models. Llama-3.1-8B does not fit on one chip.

- **Large layers make FORGE faster.** Against fused AdamW, the step time decreases by {{mlp.forge_bf16.dstep_best:.0f}}% on the MLP-1.1B (large layers, 4096 tokens per step) and by {{llama.forge_bf16.dstep_best:.0f}}% on Llama-3.2-1B (smaller layers, 512 tokens per step).
- **Large models make FORGE save more memory.** Against fused AdamW, FORGE saves G − ΔA. G is the weight gradient that FORGE deletes. ΔA is the activations that are in memory at the peak of FORGE but not at the peak of fused AdamW. For the same number of tokens, a larger model has a larger G, but its activations stay small :
  - Llama-3.1-8B (calculated) : {{pool.8b:.1f}} − {{pool_dA.8b:.1f}} = {{pool_saved.8b:.1f}} GiB saved, the peak decreases from {{8b.adamw_fused.peak:.1f}} to {{8b.forge_bf16.peak:.1f}} GiB (−{{8b.forge_bf16.dmem_best:.0f}}%).
  - Llama-3.2-1B : {{pool.1b:.4f}} − {{pool_dA.1b:.4f}} = {{pool_saved.1b:.4f}} GiB saved (−{{llama.forge_bf16.dmem_best:.0f}}%).
  - Qwen3-0.6B : −{{qwen.forge_bf16.dmem_best:.1f}}% only ([see why](#-every-model)).
- **8 chips :** I could not make the 8 chips of a Kaggle TPU v5e-8 work efficiently in the available time. The Llama-3.1-8B results below are the results of that run.

## 🦙 Llama-3.1-8B (TPU v5e-8)

Llama-3.1-8B ran on the 8 chips of a Kaggle TPU v5e-8, with tensor parallelism. This run was difficult to set up, and a Kaggle TPU v5e-8 has a waiting time of 30 to 40 hours. Thus there is only one configuration : batch 1, sequence {{seq_8b}}, one seed, 200 steps (the first 10 not timed, no lr warmup). The code is `benchmarks/kaggle/`. These numbers come from a slightly earlier version of `src/`. A new run with the current `src/` can give different numbers.

<div align="center">
<img src="assets/llama8b_headline.png" width="94%" alt="Llama-3.1-8B : peak memory of one device (calculated) and memory vs. speed">
</div>

- **The memory is calculated, not measured.** The peak memory of each chip was measured. A script converts it to the memory of one device. Each chip holds 1/8 of the weights, the optimizer states and the weight gradient : the script multiplies them by 8. Most activations on each chip are copies that all 8 chips hold (for example the residual stream). The script counts these copies one time : it multiplies the activations by {{k_1dev:.2f}}, not by 8.
- **The step time is slow.** I could not make the 8 chips work together efficiently. They give {{8b.forge_bf16.tflops:.0f}} TF/s together, {{8b.forge_bf16.mfu:.0f}}% of their peak. Use these times only to compare the methods. On this setup, the two FORGE methods are the fastest.

{{table.8b}}

## 📊 Key results (Llama-3.2-1B, one TPU v5e chip)

<div align="center">
<img src="assets/llama_headline.png" width="94%" alt="Llama-3.2-1B : peak memory and memory vs. speed">
</div>

`transformers` Llama-3.2-1B (1.24B parameters), one TPU v5e chip (16 GiB HBM, 197 TF/s bf16), batch 1, sequence 512, bf16 weights. Step time : the median of {{llama.adamw.timed}} timed steps. Each value is the mean of 2 seeds (± half the difference between the seeds).

{{table.llama}}

**FORGE (bf16) uses less memory and less time than fused AdamW.**
- Fused AdamW has its peak at the end of the backward pass, when all weight gradients are in memory.
- FORGE keeps no weight gradients, thus its peak moves to the start of the backward pass. At the end of the backward pass, FORGE holds {{llama.peak_below_fused:.2f}} GiB less than fused AdamW.
- The weight gradient that stays ({{llama.forge_bf16.grads:.2f}} GiB) belongs to the tied embedding and the norms. PyTorch updates them, not FORGE.
- The decrease is smaller than on Llama-3.1-8B because the model is small. The activations are {{act_per_weight.1b:.2f}}× the size of the weights here, and {{act_per_weight.8b:.2f}}× on Llama-3.1-8B (calculated). Thus on Llama-3.1-8B the peak decreases by almost all of the deleted gradient ({{pool_pct.8b:.0f}}%).

## ⚡ Every model

<div align="center">
<img src="assets/glance.png" width="94%" alt="Memory saved and speed-up against vanilla AdamW">
</div>

{{table.models}}

<div align="center">
<img src="assets/mlp_headline.png" width="49%" alt="MLP-1.1B"> <img src="assets/qwen_headline.png" width="49%" alt="Qwen3-0.6B">
</div>

**Qwen3-0.6B gets only a small improvement :**
- **Memory.** For each method, the peak is at the start of the backward pass. There, the activations and the logits over the {{vocab.qwen:,}}-token vocabulary are in memory, and no weight gradient exists yet. FORGE holds {{qwen.bwd_end_saved:.2f}} GiB less at the end of the backward pass, but that is not the peak. Against fused AdamW, the peak decreases by {{qwen.forge_bf16.dmem_best:.1f}}% only. With int8 moments it decreases by {{qwen.forge_int8.dmem_best:.1f}}%, because the moments are in memory at the peak.
- **Time.** Qwen3-0.6B has {{prof.qwen.forge_bf16.updates}} small weight matrices. With FORGE, the backward pass takes {{qwen.bwd_extra:.1f}} ms more than with fused AdamW, and the updates use {{prof.qwen.forge_bf16.py:.1f}} ms of Python time. The optimizer step takes only {{qwen.opt_saved:.1f}} ms less. Thus the step time increases by {{qwen.forge_bf16_slower:.1f}}%.
- **A larger Qwen3 model** has larger weight matrices and a larger weight gradient, but the same loss memory. Thus we expect better results on it. It was not run.

## ⏱️ Where the time goes

`benchmarks/profiling.py`, seed 0, median of 20 steps, ms. Forward, backward, optimizer step : a sync after each phase. Step, as trained : one sync at the end, as in the benchmark.

{{table.profile}}

## 📉 Convergence

<div align="center">
<img src="assets/convergence.png" width="96%" alt="The training loss of each step, each AdamW method, three models">
</div>

Each curve is the loss of each step : the mean of 2 seeds, without smoothing. Each model trains on 8 fixed batches.

- **FORGE (bf16) follows PyTorch.** The mean distance between its loss and the loss of fused AdamW, step by step : {{conv.mlp.forge_bf16:.4f}} (MLP-1.1B), {{conv.llama.forge_bf16:.4f}} (Llama-3.2-1B), {{conv.qwen.forge_bf16:.3f}} (Qwen3-0.6B). For vanilla AdamW it is {{conv.mlp.adamw:.4f}}, {{conv.llama.adamw:.4f}} and {{conv.qwen.adamw:.3f}}.
- **int8 moments converge more slowly.** The loss reaches the middle of its decrease (on the log scale of the figure) at step {{conv.mlp.below_half_int8}} instead of {{conv.mlp.below_half_bf16}} on the MLP-1.1B, and at step {{conv.llama.below_half_int8}} instead of {{conv.llama.below_half_bf16}} on Llama-3.2-1B. The final loss is also higher.
- **Qwen3-0.6B does not converge in these steps.** Each method has loss spikes, and each method stops near 6.0.

The final loss is the mean over the last 10% of the steps that all methods ran, and over the 2 seeds :

{{table.loss}}

## 🌀 Muon

Muon calculates `B = 0.95 B + G` for each weight. This step is local to each tile, thus FORGE does it in the dW tile. Newton–Schulz needs the full matrix, thus it runs after the backward pass. Standard Muon gets `G` from autograd and calculates `B` in a separate pass. One seed.

{{table.muon}}

On Qwen3-0.6B the peak decreases by only {{muon.qwen.forge_muon.dmem:.1f}}%, for the same cause as with AdamW.

## 🧠 How it works on TPU

<div align="center">
<img src="assets/forge_tiling.png" width="94%" alt="FORGE on TPU : tile-by-tile fused backward + optimizer in VMEM">
</div>

- **One Pallas kernel**, grid `(V/b_v, H/b_h, BT/b_bt)`. The MXU adds the gradient of each tile in fp32. Then the optimizer updates the tile.
- **In place.** `W` and the moments go to the kernel as donated buffers (`input_output_aliases`). The kernel writes the new values into the same HBM. It calculates `dx = ΔY·W` first, with the old `W`.
- **Optimizers.** AdamW (bf16 or int8 moments), SGD and Muon use the same kernel (`src/forge_tpu/kernel/`). PyTorch's fused AdamW updates the embeddings, the tied head and the norms.
- **Reproduce.** Paste `benchmarks/colab_cell.py` into one Colab cell (TPU v5e-1) and upload the repository zip. `benchmarks/report.py` writes all tables (`results/summary.md`), figures (`assets/`) and numbers of this README from `results/`. For Llama-3.1-8B : `benchmarks/kaggle/kaggle_cell.py` on a Kaggle TPU v5e-8 (see `benchmarks/kaggle/README.md`).

## ✅ Correctness

`benchmarks/tests.py` compares FORGE with `torch.optim.AdamW` (fp32, on the CPU) : **{{tests.pass}} of {{tests.total}} checks pass** ([report](results/adamw/tests.md)). The {{tests.fails}} other checks are expected. None of them is an error in the FORGE (bf16) update :
- **{{tests.int8_fails}} int8 checks.** int8 moments are approximate by design. {{tests.int8_ulp}} of these checks use the tolerance of exact bf16 values (1 ulp). One is the `v` moment with very small gradients (× 10⁻⁶), where int8 loses precision. One is a small Llama trained with int8 moments : its loss differs from the fp32 reference by up to {{tests.int8_model_err:.3f}}, against a tolerance of {{tests.int8_model_tol:.3f}} (2 × the error of PyTorch bf16).
- **1 bias check.** The bias gradient `db` has an error of {{tests.db_err:.4f}} against a tolerance of {{tests.db_tol:.3f}}. It is a bf16 sum, and FORGE does not update the bias.
