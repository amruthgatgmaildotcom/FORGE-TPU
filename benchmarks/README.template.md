<div align="center">

# 🔥 FORGE · TPU

### Fused On-Register Gradient Elimination on Google TPU v5e

*The weight gradient never reaches HBM — and `optimizer.step()` has nothing left to do.*

[![TPU v5e](https://img.shields.io/badge/Google-TPU_v5e-4285F4.svg?style=for-the-badge&logo=google&logoColor=white)](https://cloud.google.com/tpu/docs/v5e)
[![JAX Pallas](https://img.shields.io/badge/JAX-Pallas-7b3fe4.svg?style=for-the-badge)](https://docs.jax.dev/en/latest/pallas/index.html)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13_·_torch__tpu-ee4c2c.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

<p align="center"><b><a href="#-highlights">Highlights</a> · <a href="#-key-results-llama-32-1b-one-tpu-v5e-chip">Llama-3.2-1B</a> · <a href="#-llama-31-8b-tpu-v5e-8">Llama-3.1-8B</a> · <a href="#-every-model">Every model</a> · <a href="#%EF%B8%8F-where-the-time-goes">Time</a> · <a href="#-muon">Muon</a> · <a href="#-how-it-works-on-tpu">How it works</a></b></p>

</div>

---

One Pallas kernel fuses the optimizer into the weight-gradient matmul of the backward pass. For every weight tile the MXU accumulates the gradient in VMEM, the optimizer updates `W`, `m`, `v` of that tile, and writes them back **into the same HBM buffers**. **The full `grad_W` is never materialised.**

## ✨ Highlights

- **{{llama.forge_bf16.x:.2f}}× faster than PyTorch AdamW on Llama-3.2-1B** (one TPU v5e chip, {{llama.adamw.step_ms:.1f}} → {{llama.forge_bf16.step_ms:.1f}} ms/step), also faster than fused AdamW ({{llama.adamw_fused.step_ms:.1f}} ms), with **{{llama.forge_bf16.dmem:.0f}}% less peak memory** ({{llama.adamw.peak:.2f}} → {{llama.forge_bf16.peak:.2f}} GiB) — **{{llama.forge_int8.dmem:.0f}}% less with int8 moments**.
- **Llama-3.1-8B : {{8b.forge_bf16.saved:.1f}} GiB less memory** — on one device FORGE needs **{{8b.forge_bf16.peak:.1f}} GiB instead of {{8b.adamw.peak:.1f}} (vanilla AdamW) and {{8b.adamw_fused.peak:.1f}} (fused AdamW)**, and **{{8b.forge_int8.peak:.1f}} GiB with int8 moments (−{{8b.forge_int8.dmem:.0f}}%, {{8b.forge_int8.saved:.1f}} GiB saved)**. Calculated from a run on 8 chips, [see below](#-llama-31-8b-tpu-v5e-8).
- **`optimizer.step()` disappears** — on Llama-3.2-1B it shrinks from **{{prof.llama.adamw.opt:.1f}} ms (vanilla) and {{prof.llama.adamw_fused.opt:.1f}} ms (fused) to {{prof.llama.forge_bf16.opt:.1f}} ms**; only the embedding and the norms are left for it.
- **Up to {{mlp.forge_bf16.mfu:.0f}}% of the chip's peak** — with large layers (MLP-1.1B) FORGE is **{{mlp.forge_bf16.x:.2f}}× faster than vanilla AdamW and {{mlp.forge_bf16.x_best:.2f}}× faster than fused AdamW**, at {{mlp.forge_bf16.tflops:.0f}} TF/s.
- **Muon too** — **{{muon.llama.forge_muon.dmem:.0f}}% less memory and {{muon.llama.forge_muon.x:.2f}}× faster** than standard Muon on Llama-3.2-1B; **{{8b_muon.forge_muon.saved:.1f}} GiB less (−{{8b_muon.forge_muon.dmem:.0f}}%)** on Llama-3.1-8B.
- **Same training, checked** — on Llama-3.2-1B and the MLP the FORGE (bf16) loss ends within **{{same.gap:.4f}} nats** of PyTorch AdamW, and on every model it is never further from it than fused AdamW's; **{{tests.pass}} / {{tests.total}}** correctness checks pass (every fail is an int8 case, approximate by design, or the bias gradient, which FORGE does not update).

## 📊 Key results (Llama-3.2-1B, one TPU v5e chip)

<div align="center">
<img src="assets/llama_headline.png" width="94%" alt="Llama-3.2-1B : peak memory and memory vs. speed">
</div>

`transformers` Llama-3.2-1B (1.24B params), one TPU v5e chip (16 GiB, 197 TF/s bf16), batch 1, sequence 512, bf16 weights. Step time : median of {{llama.adamw.timed}} timed steps, mean of 2 seeds (± half the gap between them).

{{table.llama}}

**FORGE (bf16) beats fused AdamW on memory and on step time at once.** Fused AdamW reaches its peak at the end of backward, when every weight gradient is alive; FORGE has no gradient pool, so its peak moves to the start of backward and it ends backward holding {{llama.peak_below_fused:.2f}} GiB less. What is left of the weight gradient belongs to the tied embedding, which stays on torch.
**int8 moments** take another {{llama.int8_more_saved:.2f}} GiB off, but the moments are unpacked and packed again in every tile : the backward takes {{llama.int8_bwd_extra:.1f}} ms longer than with bf16 moments, so int8 runs at {{llama.forge_int8.x_best:.2f}}× the speed of fused AdamW.

## 🦙 Llama-3.1-8B (TPU v5e-8)

Llama-3.1-8B does not fit on one 16 GiB chip, so it ran tensor parallel on the 8 chips of a TPU v5e-8 (batch 1, sequence {{seq_8b}}, one seed).

- **Memory is calculated, not measured.** The peak of each chip was measured, then converted to one device : weights, optimizer states and weight gradient × 8 (each chip holds exactly 1/8 of them), and the activations that every chip holds a copy of are counted once (× 2.64 instead of × 8 : the ratio of one-device to one-chip activations, from a one-device run of the same model and data on another accelerator).
- **Step time is slow** : I could not make the 8 chips work together efficiently — they reach {{8b.forge_bf16.tflops:.0f}} TF/s together, {{8b.forge_bf16.mfu:.0f}}% of their peak. The times compare the methods with each other on the same setup, where both FORGE arms are the fastest.

{{table.8b}}

## ⚡ Every model

<div align="center">
<img src="assets/glance.png" width="94%" alt="Memory saved and speed-up against vanilla AdamW">
</div>

{{table.models}}

<div align="center">
<img src="assets/mlp_headline.png" width="49%" alt="MLP-1.1B"> <img src="assets/qwen_headline.png" width="49%" alt="Qwen3-0.6B">
</div>

**Why Qwen3-0.6B gains little.** Its peak is at the start of backward, in the loss over its {{vocab.qwen:,}}-token vocabulary, before any weight gradient exists — for every method. FORGE ends backward holding {{qwen.bwd_end_saved:.2f}} GiB less, but that is not the peak, so only int8 moments lower it. Its 196 weight matrices are small : FORGE's backward takes {{qwen.bwd_extra:.1f}} ms longer (it includes {{prof.qwen.forge_bf16.py:.1f}} ms of Python for the 196 updates), more than the {{qwen.opt_saved:.1f}} ms it saves in the optimizer step.

## ⏱️ Where the time goes

One step with a sync after each phase (`benchmarks/profiling.py`, median of 20 steps, ms).

{{table.profile}}

## 📉 Convergence parity

<div align="center">
<img src="assets/convergence.gif" width="78%" alt="FORGE tracks PyTorch AdamW on Llama-3.2-1B">
</div>

Final loss (mean over the last 10% of the steps that every method ran, and over the 2 seeds) :

{{table.loss}}

## 🌀 Muon

Only `B = 0.95 B + G` is tile-local, so only it is fused into the dW tile; Newton–Schulz needs the whole matrix and runs after backward. Standard Muon gets `G` from autograd and folds it into `B` in its own pass. One seed.

{{table.muon}}

On Qwen3-0.6B the memory does not change, for the same reason as with AdamW : its peak is at the start of backward.

## 🧠 How it works on TPU

<div align="center">
<img src="assets/forge_tiling.png" width="94%" alt="FORGE on TPU : tile-by-tile fused backward + optimizer in VMEM">
</div>

- **One Pallas kernel**, grid `(V/b_v, H/b_h, BT/b_bt)` : the MXU accumulates the gradient tile in fp32, then the optimizer runs on the tile. `W` and the moments are donated (`input_output_aliases`), so the update is in place; `dx = ΔY·W` is issued first, with the old `W`.
- **AdamW (bf16 or int8 moments), SGD and Muon** share the kernel (`src/forge_tpu/kernel/`). Embeddings, the tied head and the norms stay on fused AdamW.
- **Reproduce** : paste `benchmarks/colab_cell.py` into one Colab cell (TPU v5e-1) and upload the repo zip; `benchmarks/report.py` writes every table (`results/summary.md`), figure (`assets/`) and number of this README from `results/`.
