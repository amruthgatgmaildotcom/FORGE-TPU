<div align="center">

# 🔥 FORGE · TPU

### Fused On-Register Gradient Elimination on Google TPU v5e

*The weight gradient does not go to HBM. The optimizer step has almost no work.*

[![TPU v5e](https://img.shields.io/badge/Google-TPU_v5e-4285F4.svg?style=for-the-badge&logo=google&logoColor=white)](https://cloud.google.com/tpu/docs/v5e)
[![JAX Pallas](https://img.shields.io/badge/JAX-Pallas-7b3fe4.svg?style=for-the-badge)](https://docs.jax.dev/en/latest/pallas/index.html)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13_·_torch__tpu-ee4c2c.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

<p align="center"><b><a href="#-highlights">Highlights</a> · <a href="#-key-results-llama-32-1b-one-tpu-v5e-chip">Llama-3.2-1B</a> · <a href="#-llama-31-8b-tpu-v5e-8">Llama-3.1-8B</a> · <a href="#-every-model">Every model</a> · <a href="#%EF%B8%8F-where-the-time-goes">Time</a> · <a href="#-muon">Muon</a> · <a href="#-how-it-works-on-tpu">How it works</a></b></p>

</div>

---

FORGE puts the optimizer step into the backward pass. One Pallas kernel calculates the weight gradient one tile at a time, in VMEM. The same kernel then updates `W`, `m` and `v` of that tile. It writes them back into the same HBM buffers. **The full weight gradient is never in HBM.**

## ✨ Highlights

All changes are against vanilla PyTorch AdamW on the same TPU, unless the text gives a different reference.

- **Llama-3.2-1B, one TPU v5e chip :** FORGE decreases the step time by **{{llama.forge_bf16.dstep:.0f}}%** ({{llama.adamw.step_ms:.1f}} → {{llama.forge_bf16.step_ms:.1f}} ms). It decreases the peak memory by **{{llama.forge_bf16.dmem:.0f}}%** ({{llama.adamw.peak:.2f}} → {{llama.forge_bf16.peak:.2f}} GiB). With int8 moments, the peak memory decreases by **{{llama.forge_int8.dmem:.0f}}%**.
- **Llama-3.2-1B, against fused AdamW :** FORGE decreases the step time by **{{llama.forge_bf16.dstep_best:.0f}}%** and the peak memory by **{{llama.forge_bf16.dmem_best:.0f}}%**. With int8 moments, the peak memory decreases by **{{llama.forge_int8.dmem_best:.0f}}%**.
- **Optimizer step, Llama-3.2-1B :** its time decreases by **{{llama.opt_cut:.0f}}%** ({{prof.llama.adamw.opt:.1f}} → {{prof.llama.forge_bf16.opt:.1f}} ms), and by **{{llama.opt_cut_best:.0f}}%** against fused AdamW.
- **Llama-3.1-8B, one device (calculated) :** FORGE decreases the peak memory by **{{8b.forge_bf16.dmem:.0f}}%** ({{8b.adamw.peak:.1f}} → {{8b.forge_bf16.peak:.1f}} GiB, {{8b.forge_bf16.saved:.1f}} GiB less). With int8 moments, the peak memory decreases by **{{8b.forge_int8.dmem:.0f}}%** ({{8b.forge_int8.peak:.1f}} GiB). [The memory is calculated from a run on 8 chips](#-llama-31-8b-tpu-v5e-8).
- **MLP-1.1B, one chip (large layers) :** FORGE decreases the step time by **{{mlp.forge_bf16.dstep:.0f}}%**, and by **{{mlp.forge_bf16.dstep_best:.0f}}%** against fused AdamW. The chip runs at **{{mlp.forge_bf16.mfu:.0f}}%** of its bf16 peak. The peak memory decreases by **{{mlp.forge_bf16.dmem:.0f}}%** ({{mlp.forge_int8.dmem:.0f}}% with int8 moments).
- **Muon :** against standard Muon on Llama-3.2-1B, FORGE decreases the peak memory by **{{muon.llama.forge_muon.dmem:.0f}}%** and the step time by **{{muon.llama.forge_muon.dstep:.0f}}%**. On Llama-3.1-8B (one device, calculated), it decreases the peak memory by **{{8b_muon.forge_muon.dmem:.0f}}%**.
- **Training is the same :** on Llama-3.2-1B and the MLP, the final loss of FORGE (bf16) is less than **{{same.gap:.4f}} nats** from the final loss of PyTorch AdamW. On each model, FORGE is not further from PyTorch AdamW than fused AdamW is. **{{tests.pass}} of {{tests.total}}** correctness checks pass. Each check that fails is an int8 check (int8 is not exact) or the bias gradient (FORGE does not update the bias).

## 📊 Key results (Llama-3.2-1B, one TPU v5e chip)

<div align="center">
<img src="assets/llama_headline.png" width="94%" alt="Llama-3.2-1B : peak memory and memory vs. speed">
</div>

Model : `transformers` Llama-3.2-1B (1.24B parameters). Hardware : one TPU v5e chip (16 GiB HBM, 197 TF/s bf16). Batch 1, sequence 512, bf16 weights. The step time is the median of {{llama.adamw.timed}} timed steps. Each value is the mean of 2 seeds (± half the difference between the seeds).

{{table.llama}}

**FORGE (bf16) uses less memory and less time than fused AdamW.**
- Fused AdamW has its peak memory at the end of the backward pass. At that point, all weight gradients are in memory.
- FORGE keeps no weight gradients. Its peak memory moves to the start of the backward pass. At the end of the backward pass, FORGE holds {{llama.peak_below_fused:.2f}} GiB less than fused AdamW.
- The weight gradient that stays (0.49 GiB) is the gradient of the tied embedding. PyTorch updates the embedding, not FORGE.

**int8 moments decrease the memory by {{llama.int8_more_saved:.2f}} GiB more, but they are slower than bf16 moments.** The kernel unpacks and packs the int8 moments in each tile. Thus the backward pass takes {{llama.int8_bwd_extra:.1f}} ms more than with bf16 moments. With int8 moments, the step time is {{llama.int8_slower_best:.1f}}% more than with fused AdamW.

## 🦙 Llama-3.1-8B (TPU v5e-8)

Llama-3.1-8B is too large for one 16 GiB chip. Thus it ran on the 8 chips of a TPU v5e-8, with tensor parallelism. Batch 1, sequence {{seq_8b}}, one seed.

- **The memory is calculated, not measured.** The peak memory of each chip was measured. A script converted it to the memory of one device. Each chip holds 1/8 of the weights, the optimizer states and the weight gradient, thus the script multiplies them by 8. Each chip also holds a copy of most activations. The script counts these copies one time : it multiplies the activations by 2.64, not by 8. The value 2.64 is the ratio of one-device to one-chip activations, from a one-device run of the same model and data on a different accelerator.
- **The step time is slow.** I could not make the 8 chips work together efficiently. The 8 chips give {{8b.forge_bf16.tflops:.0f}} TF/s together, {{8b.forge_bf16.mfu:.0f}}% of their peak. Use these times only to compare the methods with each other. On this setup, the two FORGE methods are the fastest.

{{table.8b}}

## ⚡ Every model

<div align="center">
<img src="assets/glance.png" width="94%" alt="Memory saved and speed-up against vanilla AdamW">
</div>

{{table.models}}

<div align="center">
<img src="assets/mlp_headline.png" width="49%" alt="MLP-1.1B"> <img src="assets/qwen_headline.png" width="49%" alt="Qwen3-0.6B">
</div>

**Qwen3-0.6B gets a small improvement only. There are two causes :**
- **Memory.** For each method, the peak memory is at the start of the backward pass. At that point, the loss over the {{vocab.qwen:,}}-token vocabulary is in memory, and no weight gradient exists yet. FORGE holds {{qwen.bwd_end_saved:.2f}} GiB less at the end of the backward pass, but that is not the peak. Only int8 moments decrease the peak.
- **Time.** Qwen3-0.6B has 196 small weight matrices. With FORGE, the backward pass takes {{qwen.bwd_extra:.1f}} ms more. Of this time, {{prof.qwen.forge_bf16.py:.1f}} ms is Python for the 196 updates. The optimizer step takes {{qwen.opt_saved:.1f}} ms less. Thus the total step time increases.

## ⏱️ Where the time goes

Each row is one step, with a sync after each phase (`benchmarks/profiling.py`, median of 20 steps, ms).

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

On Qwen3-0.6B the memory does not change. The cause is the same as with AdamW : the peak memory is at the start of the backward pass.

## 🧠 How it works on TPU

<div align="center">
<img src="assets/forge_tiling.png" width="94%" alt="FORGE on TPU : tile-by-tile fused backward + optimizer in VMEM">
</div>

- **One Pallas kernel** with the grid `(V/b_v, H/b_h, BT/b_bt)`. The MXU adds the gradient of each tile in fp32. Then the optimizer updates the tile.
- **In place.** The kernel gets `W` and the moments as donated buffers (`input_output_aliases`). Thus it writes the new values into the same HBM. The kernel calculates `dx = ΔY·W` first, with the old `W`.
- **Optimizers.** AdamW (bf16 or int8 moments), SGD and Muon use the same kernel (`src/forge_tpu/kernel/`). PyTorch's fused AdamW updates the embeddings, the tied head and the norms.
- **Reproduce.** Paste `benchmarks/colab_cell.py` into one Colab cell (TPU v5e-1). Upload the repository zip. `benchmarks/report.py` writes all tables (`results/summary.md`), all figures (`assets/`) and all numbers of this README from `results/`.
