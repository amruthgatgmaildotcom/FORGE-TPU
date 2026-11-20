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

FORGE puts the optimizer step into the backward pass. One Pallas kernel calculates the weight gradient one tile at a time, in VMEM. The same kernel then updates `W`, `m` and `v` of that tile. It writes them back into the same HBM buffers. **The full weight gradient is never in HBM.**

## ✨ Highlights

All changes are against vanilla PyTorch AdamW on the same TPU, unless the text gives a different reference.

- **Llama-3.1-8B, one device :** FORGE decreases the peak memory by **{{8b.forge_bf16.dmem:.0f}}%** ({{8b.adamw.peak:.1f}} → {{8b.forge_bf16.peak:.1f}} GiB, {{8b.forge_bf16.saved:.1f}} GiB less). With int8 moments, the peak memory decreases by **{{8b.forge_int8.dmem:.0f}}%** ({{8b.forge_int8.peak:.1f}} GiB). [More details](#-llama-31-8b-tpu-v5e-8).
- **Less memory :** FORGE decreases the peak memory on each model.
  - Llama-3.2-1B : **−{{llama.forge_bf16.dmem:.0f}}%** ({{llama.adamw.peak:.2f}} → {{llama.forge_bf16.peak:.2f}} GiB), **−{{llama.forge_int8.dmem:.0f}}%** with int8 moments. Against fused AdamW : −{{llama.forge_bf16.dmem_best:.0f}}%, −{{llama.forge_int8.dmem_best:.0f}}% with int8 moments.
  - MLP-1.1B : **−{{mlp.forge_bf16.dmem:.0f}}%** ({{mlp.adamw.peak:.2f}} → {{mlp.forge_bf16.peak:.2f}} GiB), **−{{mlp.forge_int8.dmem:.0f}}%** with int8 moments. Against fused AdamW : −{{mlp.forge_bf16.dmem_best:.0f}}%, −{{mlp.forge_int8.dmem_best:.0f}}% with int8 moments.
  - Muon, against standard Muon : **−{{muon.llama.forge_muon.dmem:.0f}}%** on Llama-3.2-1B, **−{{8b_muon.forge_muon.dmem:.0f}}%** on Llama-3.1-8B (one device, calculated).
- **Less time :** FORGE decreases the step time on each model of one chip. FORGE (bf16) is also faster than fused AdamW, except on Qwen3-0.6B ([see why](#-every-model)).
  - Llama-3.2-1B : **−{{llama.forge_bf16.dstep:.0f}}%** ({{llama.adamw.step_ms:.1f}} → {{llama.forge_bf16.step_ms:.1f}} ms). Against fused AdamW : −{{llama.forge_bf16.dstep_best:.0f}}%.
  - MLP-1.1B : **−{{mlp.forge_bf16.dstep:.0f}}%** ({{mlp.adamw.step_ms:.1f}} → {{mlp.forge_bf16.step_ms:.1f}} ms). Against fused AdamW : −{{mlp.forge_bf16.dstep_best:.0f}}%. The chip runs at **{{mlp.forge_bf16.mfu:.0f}}%** of its bf16 peak.
  - The optimizer step of Llama-3.2-1B : **−{{llama.opt_cut:.0f}}%** ({{prof.llama.adamw.opt:.1f}} → {{prof.llama.forge_bf16.opt:.1f}} ms). Against fused AdamW : −{{llama.opt_cut_best:.0f}}%.
  - Muon, against standard Muon : **−{{muon.llama.forge_muon.dstep:.0f}}%** on Llama-3.2-1B.
- **The main limit is the HBM, not FORGE.** One TPU v5e chip has only 16 GiB of HBM. [See below](#%EF%B8%8F-the-main-limit--16-gib-of-hbm).

## ⚠️ The main limit : 16 GiB of HBM

The largest disadvantage of this work is the small HBM of one TPU v5e chip (16 GiB). FORGE gives the most improvement on large layers and on large models. A 16 GiB chip cannot hold these models.

- **Large layers make FORGE faster.** Against fused AdamW, FORGE decreases the step time by {{mlp.forge_bf16.dstep_best:.0f}}% on the MLP-1.1B (large layers, 4096 tokens per step, {{mlp.forge_bf16.mfu:.0f}}% of the chip's peak). On Llama-3.2-1B (smaller layers, 512 tokens per step), it decreases the step time by {{llama.forge_bf16.dstep_best:.0f}}%.
- **Large models make FORGE save more memory.** On Llama-3.2-1B, the peak memory decreases by {{pool_pct.1b:.0f}}% of the weight gradient that FORGE deletes ({{pool_saved.1b:.2f}} of {{pool.1b:.2f}} GiB). On Llama-3.1-8B (calculated), it decreases by {{pool_pct.8b:.0f}}% of it ({{pool_saved.8b:.1f}} of {{pool.8b:.1f}} GiB).
- **On 16 GiB, a large model must be split.** Llama-3.1-8B needs 8 chips. Each chip then holds only 1/8 of each layer, and the chips must communicate. This makes the step slow (see [Llama-3.1-8B](#-llama-31-8b-tpu-v5e-8)).

On a TPU with more HBM, a large model fits on one chip. There, we expect FORGE to give much larger improvements in memory and in time. This is not measured yet.

## 🦙 Llama-3.1-8B (TPU v5e-8)

Llama-3.1-8B is too large for one 16 GiB chip. Thus it ran on the 8 chips of a Kaggle TPU v5e-8, with tensor parallelism. This run was difficult to set up, and a Kaggle TPU v5e-8 has a waiting time of 30 to 40 hours. Thus there is only one configuration : batch 1, sequence {{seq_8b}}, one seed. The run used an earlier version of this code (the `kaggle/` folder of the `python-old` branch, without the 20 lr-warmup steps).

<div align="center">
<img src="assets/llama8b_headline.png" width="94%" alt="Llama-3.1-8B : peak memory of one device (calculated) and memory vs. speed">
</div>

- **The memory is calculated, not measured.** The peak memory of each chip was measured. A script converted it to the memory of one device. Each chip holds 1/8 of the weights, the optimizer states and the weight gradient, thus the script multiplies them by 8. Each chip also holds a copy of most activations. The script counts these copies one time : it multiplies the activations by {{k_1dev:.2f}}, not by 8.
- **The step time is slow.** I could not make the 8 chips work together efficiently. The 8 chips give {{8b.forge_bf16.tflops:.0f}} TF/s together, {{8b.forge_bf16.mfu:.0f}}% of their peak. Use these times only to compare the methods with each other. On this setup, the two FORGE methods are the fastest.

{{table.8b}}

## 📊 Key results (Llama-3.2-1B, one TPU v5e chip)

<div align="center">
<img src="assets/llama_headline.png" width="94%" alt="Llama-3.2-1B : peak memory and memory vs. speed">
</div>

Model : `transformers` Llama-3.2-1B (1.24B parameters). Hardware : one TPU v5e chip (16 GiB HBM, 197 TF/s bf16). Batch 1, sequence 512, bf16 weights. The step time is the median of {{llama.adamw.timed}} timed steps. Each value is the mean of 2 seeds (± half the difference between the seeds).

{{table.llama}}

**FORGE (bf16) uses less memory and less time than fused AdamW.**
- Fused AdamW has its peak memory at the end of the backward pass. At that point, all weight gradients are in memory.
- FORGE keeps no weight gradients. Its peak memory moves to the start of the backward pass. At the end of the backward pass, FORGE holds {{llama.peak_below_fused:.2f}} GiB less than fused AdamW.
- The weight gradient that stays ({{llama.forge_bf16.grads:.2f}} GiB) is the gradient of the tied embedding and of the norms. PyTorch updates them, not FORGE.
- The memory decrease looks small only because the model is small. At the peak of FORGE, the activations are in memory. On Llama-3.2-1B, the activations are {{act_per_weight.1b:.2f}}× the size of the weights. On Llama-3.1-8B, they are {{act_per_weight.8b:.2f}}× (calculated). Thus on Llama-3.1-8B, the peak decreases by almost all of the deleted weight gradient ({{pool_pct.8b:.0f}}%). This is almost the maximum decrease that deleting the weight gradient can give.

## ⚡ Every model

<div align="center">
<img src="assets/glance.png" width="94%" alt="Memory saved and speed-up against vanilla AdamW">
</div>

{{table.models}}

<div align="center">
<img src="assets/mlp_headline.png" width="49%" alt="MLP-1.1B"> <img src="assets/qwen_headline.png" width="49%" alt="Qwen3-0.6B">
</div>

**Qwen3-0.6B gets a small improvement only. There are two causes :**
- **Memory.** For each method, the peak memory is at the start of the backward pass. At that point, the activations and the logits over the {{vocab.qwen:,}}-token vocabulary are in memory, and no weight gradient exists yet. FORGE holds {{qwen.bwd_end_saved:.2f}} GiB less at the end of the backward pass, but that is not the peak. Thus FORGE (bf16) decreases the peak by only {{qwen.forge_bf16.dmem_best:.1f}}% against fused AdamW. int8 moments decrease it by {{qwen.forge_int8.dmem_best:.1f}}%, because the moments are in memory at the peak and int8 moments are smaller.
- **Time.** Qwen3-0.6B has {{prof.qwen.forge_bf16.updates}} small weight matrices. With FORGE, the backward pass takes {{qwen.bwd_extra:.1f}} ms more than with fused AdamW, and the {{prof.qwen.forge_bf16.updates}} updates use {{prof.qwen.forge_bf16.py:.1f}} ms of Python time. The optimizer step takes only {{qwen.opt_saved:.1f}} ms less. Thus the total step time increases by {{qwen.forge_bf16_slower:.1f}}%.
- **A larger Qwen3 model.** A larger model has larger weight matrices, and a larger weight gradient. The memory of the loss stays the same. Thus we expect FORGE to give better memory and time on a larger Qwen3 model. A larger Qwen3 does not fit in 16 GiB with vanilla AdamW, thus it is not measured.

## ⏱️ Where the time goes

`benchmarks/profiling.py`, seed 0, median of 20 steps, ms. Forward, backward and optimizer step : a sync after each phase. Step, as trained : one sync at the end of the step, as in the benchmark.

{{table.profile}}

## 📉 Convergence parity

<div align="center">
<img src="assets/convergence.gif" width="78%" alt="FORGE follows PyTorch AdamW on Llama-3.2-1B">
</div>

The final loss is the mean over the last 10% of the steps that all methods ran, and over the 2 seeds :

{{table.loss}}

## 🌀 Muon

Muon calculates `B = 0.95 B + G` for each weight. This step is local to each tile, thus FORGE does it in the dW tile. Newton–Schulz needs the full matrix, thus it runs after the backward pass. Standard Muon gets `G` from autograd and calculates `B` in a separate pass. One seed.

{{table.muon}}

On Qwen3-0.6B the peak memory decreases by only {{muon.qwen.forge_muon.dmem:.1f}}%. The cause is the same as with AdamW : the peak memory is at the start of the backward pass.

## 🧠 How it works on TPU

<div align="center">
<img src="assets/forge_tiling.png" width="94%" alt="FORGE on TPU : tile-by-tile fused backward + optimizer in VMEM">
</div>

- **One Pallas kernel** with the grid `(V/b_v, H/b_h, BT/b_bt)`. The MXU adds the gradient of each tile in fp32. Then the optimizer updates the tile.
- **In place.** The kernel gets `W` and the moments as donated buffers (`input_output_aliases`). Thus it writes the new values into the same HBM. The kernel calculates `dx = ΔY·W` first, with the old `W`.
- **Optimizers.** AdamW (bf16 or int8 moments), SGD and Muon use the same kernel (`src/forge_tpu/kernel/`). PyTorch's fused AdamW updates the embeddings, the tied head and the norms.
- **Reproduce.** Paste `benchmarks/colab_cell.py` into one Colab cell (TPU v5e-1). Upload the repository zip. `benchmarks/report.py` writes all tables (`results/summary.md`), all figures (`assets/`) and all numbers of this README from `results/`. The Llama-3.1-8B results come from the `kaggle/` code of the `python-old` branch.
