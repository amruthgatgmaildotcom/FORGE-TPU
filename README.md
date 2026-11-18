<div align="center">

# 𓉱 FORGE TPU 𓉱

### FORGE (Fused On-Register Gradient Elimination) on Google TPU v5e

</div>
<br>
<br>
<br>


## HEADING-1

- `FusedLinear` (`src/forge_tpu/module.py`) is a drop in replacement for `nn.Linear`. it also holds the optimizer details of that layer (design choice), `FusedLinear.from_linear` wraps an existing `nn.Linear` without copying the weight.
- in backward, `dx = go @ w` is computed first with the old w, then the Pallas kernel (`src/forge_tpu/kernel/fused_grad.py`) runs over a grid of (bv, bh) weight tiles. for each tile it accumulates `go.T @ x` in an fp32 VMEM scratch pad, then reads the w, m, v tile, does the AdamW math and writes them back.
- w and the states are donated to the kernel (`input_output_aliases`), so XLA writes the new values into the same HBM. torch_tpu marks donated tensors as deleted, that is why `_rebind` has to `set_` the weight and the state tensors to the outputs.
- torch tensors reach the jax kernel through `torch.tpu.pallas.jax_op`. its python checks cost about 160us per layer, so `_bridge.py` catches the first launch of every shape and replays it directly after that (about 21us).
- int8 states are int8 + one fp32 scale per 64 rows (`kernel/codecs.py`). to add another optimizer or state format, see the comments at the top of `kernel/optimizers.py` and `kernel/codecs.py`.

## HEADING-2

| Name | .... |
|---|---|
| vanilla AdamW | `torch.optim.AdamW()` (torch's defaults), bf16 weights, bf16 states |
| fused AdamW | `torch.optim.AdamW(fused=True)`, bf16 weights, bf16 states |
| FORGE (bf16) | FORGE, bf16 weights, bf16 states |
| FORGE (int8) | FORGE, bf16 weights, int8 states |

## Models

all models use random weights and random data, no weights are downloaded.

- Llama-3.2-1B and Qwen3-0.6B : the model code is from the `transformers` library (`LlamaForCausalLM`, `Qwen3ForCausalLM`) with the numbers of the official `config.json`. every Linear inside the layers is FORGE, the tied embedding / lm_head and the norms stay on torch's fused AdamW.
- MLP : a residual MLP, defined in `benchmarks/mlp.py`. everything in it is FORGE.

| Model | Parameters | Tokens per step |
|---|---|---|
| MLP | 1.09B | 4096 |
| Llama-3.2-1B | 1.24B | 1024 |
| Qwen3-0.6B | 0.60B | 1024 |



## Results (TPU v5e-1)

every number and plot below comes from `results/adamw/` : `benchmarks/readme.py` writes them into this file (the blocks between `<!-- auto -->` comments), so do not edit them by hand.
step time is the median of the timed steps (± std), the first 10 steps are not timed. each run stops at 300 steps or after 60 s, that is why the number of steps is different
(Qwen : 1000 steps, from a separate benchmark-only run, because 60 s was too noisy). the caption above each table gives the number of timed steps.

for each model there are three plots :

1. memory right before `backward()` and when it ends (weights, states, activations, weight gradient), and memory vs speed.
2. loss at each training step.
3. memory at each step, and memory inside one step, layer by layer.

### MLP

<!-- auto:results adamw/mlp -->
MLP, 1.09B params, adamw. Step time : median (± std) of 213 to 260 timed steps of each run, after 10 warmup steps. Peak : the largest step peak above the data.

| Method | Peak (GiB) | Step (ms) | TF/s |
|---|:--:|:--:|:--:|
| vanilla AdamW | 13.91 | 259.6 ± 0.2 | 103 ± 0 |
| fused AdamW | 12.69 | 216.8 ± 0.2 | 124 ± 0 |
| FORGE (bf16) | 10.67 | 187.8 ± 0.1 | 143 ± 0 |
| FORGE (int8) | 8.76 | 189.0 ± 0.2 | 142 ± 0 |

| Method | Loss at step 223 (last common step) | max \|loss − vanilla AdamW\| (steps 1 to 223) |
|---|:--:|:--:|
| vanilla AdamW | 0.3941 | 0.0000 |
| fused AdamW | 0.3941 | 0.0004 |
| FORGE (bf16) | 0.3941 | 0.0004 |
| FORGE (int8) | 0.4028 | 0.3211 |

![MLP headline](results/adamw/mlp/mlp_headline.png)
![MLP loss](results/adamw/mlp/mlp_loss.png)
![MLP memory](results/adamw/mlp/mlp_memory.png)
<!-- /auto -->

### Llama-3.2-1B

<!-- auto:results adamw/llama -->
Llama-3.2-1B, 1.24B params, adamw. Step time : median (± std) of 208 to 256 timed steps of each run, after 10 warmup steps. Peak : the largest step peak above the data.

| Method | Peak (GiB) | Step (ms) | TF/s |
|---|:--:|:--:|:--:|
| vanilla AdamW | 12.04 | 226.1 ± 10.9 | 35 ± 2 |
| fused AdamW | 11.37 | 183.7 ± 20.1 | 44 ± 5 |
| FORGE (bf16) | 11.31 | 159.7 ± 15.7 | 50 ± 5 |
| FORGE (int8) | 9.57 | 159.1 ± 19.5 | 50 ± 6 |

| Method | Loss at step 218 (last common step) | max \|loss − vanilla AdamW\| (steps 1 to 218) |
|---|:--:|:--:|
| vanilla AdamW | 0.0426 | 0.0000 |
| fused AdamW | 0.0387 | 8.9854 |
| FORGE (bf16) | 0.0381 | 10.5057 |
| FORGE (int8) | 0.2533 | 7.5098 |

![Llama-3.2-1B headline](results/adamw/llama/llama_headline.png)
![Llama-3.2-1B loss](results/adamw/llama/llama_loss.png)
![Llama-3.2-1B memory](results/adamw/llama/llama_memory.png)
<!-- /auto -->

### Qwen3-0.6B

<!-- auto:results adamw/qwen -->
Qwen3-0.6B, 0.60B params, adamw. Step time : median (± std) of 990 timed steps of each run, after 10 warmup steps. Peak : the largest step peak above the data.

| Method | Peak (GiB) | Step (ms) | TF/s |
|---|:--:|:--:|:--:|
| vanilla AdamW | 8.14 | 394.0 ± 43.7 | 11 ± 1 |
| fused AdamW | 8.11 | 362.3 ± 39.3 | 12 ± 1 |
| FORGE (bf16) | 8.10 | 367.7 ± 36.3 | 12 ± 1 |
| FORGE (int8) | 7.33 | 371.0 ± 38.8 | 12 ± 1 |

| Method | Loss at step 1000 (last common step) | max \|loss − vanilla AdamW\| (steps 1 to 1000) |
|---|:--:|:--:|
| vanilla AdamW | 2.5772 | 0.0000 |
| fused AdamW | 6.0923 | 4.4498 |
| FORGE (bf16) | 0.2282 | 4.0127 |
| FORGE (int8) | 5.9495 | 12.6477 |

![Qwen3-0.6B headline](results/adamw/qwen/qwen_headline.png)
![Qwen3-0.6B loss](results/adamw/qwen/qwen_loss.png)
![Qwen3-0.6B memory](results/adamw/qwen/qwen_memory.png)
<!-- /auto -->

TF/s = model FLOPs of one step (6 x params x tokens, + attention) / step time. the v5e peak is 197 TF/s in bf16.

## IMPORTANT

- MLP : big layers, so the TPU is busy the whole step. here FORGE wins clearly.
- Llama and Qwen : small layers and lots of tiny ops. we can roughly say params per layer higher is where forge performs better than fused, it is better than vanilla any day. if layer width is more but no of layers are less then tpu time dominates over host time and if its the other way around then host time dominates. we want tpu time to dominate since tpu and host run parallel at times and FORGE improves tpu time only.
- peak memory : right before backward, we have outputs of previous layer stored into ctx, we have a dx which keeps mutating for next layer in .backward() (this is negligible anyways), we
  have state tensors. and during backward, fused (torch) keeps accumilating new dw (V, H) but also keeps discarding the prev layer outputs (BT, D). let G = total space dw takes and let A = 
  total space all prev layer outputs take. relative to torch, FORGE saves max(0, G - A) (peak memory). for small models (current case) say llama-1B, G=2.3GiB, A=2.2GiB. but for llama-8B, 
  G=15GiB and A=9GiB (both are for 1024 seq_len). hence the saving dont appear significant right now, but for bigger models FORGE must be performing noticably better (both time and space).
- vanilla vs fused AdamW on Llama : their losses drift apart a lot after ~60 steps (the model memorizes 8 fixed sequences and that part is very sensitive to rounding),
  so compare FORGE with fused AdamW there, not with vanilla. (cuz FORGE and fused do kernel math in fp32, but vanilla bf16 does in bf16)

## Correctness

`benchmarks/tests.py` compares FORGE against `torch.optim.AdamW`. (fully AI generated code)
- 153 / 160 checks pass. the 7 that fail are the int8 ones (int8 is not exact, thats expected) and the bias `db` check (bf16 sum, FORGE doesnt touch the bias anyway).
- FORGE (bf16) gives the same result as PyTorch : each weight and state is within 1 bf16 ulp.
- on the MLP, the FORGE (bf16) loss and the fused AdamW loss are almost the same (max difference <!-- auto:lossdiff adamw/mlp forge_bf16 adamw_fused -->0.0002<!-- /auto -->).

## Tiles

the kernel uses one tile size `(bv, bh, bbt)` for every layer (`DEFAULT_TILES` in `src/forge_tpu/kernel/_utils.py`, right now `(512, 2048, 2048)`).
`benchmarks/colab_cell_tiler.py` times many tile sizes on every fused layer shape of the three models and reports the fastest. it changes nothing. (fully AI generated code)
(512, 2048, 2048) performed best as one triple for all three models, for AdamW and SGD (against the old default (512, 1024, 1024) : MLP 0.88 / 0.95, Llama 0.89 / 0.88, Qwen 1.04 / 1.05 of the time). the results above ran with it.

## Profile

- `benchmarks/profiling.py` shows where the time of one training step goes, for the same models and methods (fully AI generated code)
- `profile_<model>_timeline.png` : one step on a time axis. for each method, the step as the benchmark runs it (when python has issued each phase, and when the TPU is done), and the same step with a sync after each phase (fully AI generated code) (every code inside benchmarks folder is claude)
- `profile_<model>_table.md` : the same numbers as a table.

### MLP

<!-- auto:profile adamw/mlp -->
#### MLP (adamw) : where the time of one step goes (ms, median of 20 steps)

| | vanilla AdamW | fused AdamW | FORGE (bf16) | FORGE (int8) |
|---|:--:|:--:|:--:|:--:|
| **Sync after each phase** | | | | |
| Forward | 68.8 | 68.7 | 68.7 | 68.7 |
| Backward : Python returns | 1.5 | 1.5 | 3.3 | 3.7 |
| Backward : TPU finishes | 123.6 | 123.5 | 118.1 | 119.2 |
| Optimizer step | 67.4 | 25.2 | 0.1 | 0.1 |
| Total | 259.8 | 217.5 | 186.8 | 188.0 |
| **As the benchmark (one sync at the end)** | | | | |
| Python done issuing | 3.5 | 2.7 | 4.8 | 5.1 |
| Step time (TPU done) | **257.8** | **215.5** | **186.2** | **187.3** |
| of which FORGE Python | - | - | 1.6 | 1.7 |
| FORGE updates per step | - | - | 17 | 17 |

#### MLP (adamw) : profile in detail (median of 20 steps, after 10 warmup steps)

| Method | Step as benchmark (ms) | Python done issuing (ms) | Forward : Python / TPU done | Backward : Python / TPU done | Step : Python / TPU done | FORGE Python (ms) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla AdamW | 257.8 | 3.5 | 1.2 / 68.8 | 1.5 / 123.6 | 1.2 / 67.4 | 0.0 |
| fused AdamW | 215.5 | 2.7 | 1.0 / 68.7 | 1.5 / 123.5 | 0.4 / 25.2 | 0.0 |
| FORGE (bf16) | 186.2 | 4.8 | 1.7 / 68.7 | 3.3 / 118.1 | 0.0 / 0.1 | 1.6 |
| FORGE (int8) | 187.3 | 5.1 | 1.6 / 68.7 | 3.7 / 119.2 | 0.0 / 0.1 | 1.7 |

Host time per phase (ms, under the profiler), the 3 largest groups :

| Method | Forward | Backward | Step |
|---|---|---|---|
| vanilla AdamW | elementwise 0.5, Python, waiting (no op) 0.4, matmul 0.2 | Python, waiting (no op) 0.6, copies, new tensors 0.5, autograd engine 0.4 | optimizer 0.8, copies, new tensors 0.1, Python, waiting (no op) 0.0 |
| fused AdamW | elementwise 0.6, Python, waiting (no op) 0.4, matmul 0.2 | Python, waiting (no op) 0.5, copies, new tensors 0.5, autograd engine 0.4 | optimizer 0.3, Python, waiting (no op) 0.0, copies, new tensors 0.0 |
| FORGE (bf16) | elementwise 0.7, Python, waiting (no op) 0.6, matmul 0.3 | FORGE update (Python) 1.4, copies, new tensors 0.9, Python, waiting (no op) 0.6 | Python, waiting (no op) 0.0 |
| FORGE (int8) | elementwise 0.6, Python, waiting (no op) 0.4, matmul 0.3 | FORGE update (Python) 1.6, copies, new tensors 0.9, Python, waiting (no op) 0.6 | Python, waiting (no op) 0.0 |

![MLP profile timeline](results/adamw/mlp/profile_mlp_timeline.png)
![MLP profile host](results/adamw/mlp/profile_mlp_host.png)
<!-- /auto -->

### Llama-3.2-1B

<!-- auto:profile adamw/llama -->
#### Llama-3.2-1B (adamw) : where the time of one step goes (ms, median of 20 steps)

| | vanilla AdamW | fused AdamW | FORGE (bf16) | FORGE (int8) |
|---|:--:|:--:|:--:|:--:|
| **Sync after each phase** | | | | |
| Forward | 61.4 | 58.0 | 54.8 | 59.4 |
| Backward : Python returns | 36.7 | 39.2 | 64.0 | 57.7 |
| Backward : TPU finishes | 92.7 | 92.1 | 92.7 | 106.5 |
| Optimizer step | 79.2 | 31.4 | 7.7 | 7.9 |
| Total | 233.3 | 181.5 | 155.1 | 173.8 |
| **As the benchmark (one sync at the end)** | | | | |
| Python done issuing | 70.2 | 76.2 | 99.5 | 95.6 |
| Step time (TPU done) | **195.1** | **176.4** | **152.1** | **164.7** |
| of which FORGE Python | - | - | 16.2 | 15.1 |
| FORGE updates per step | - | - | 112 | 112 |

#### Llama-3.2-1B (adamw) : profile in detail (median of 20 steps, after 10 warmup steps)

| Method | Step as benchmark (ms) | Python done issuing (ms) | Forward : Python / TPU done | Backward : Python / TPU done | Step : Python / TPU done | FORGE Python (ms) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla AdamW | 195.1 | 70.2 | 28.2 / 61.4 | 36.7 / 92.7 | 9.5 / 79.2 | 0.0 |
| fused AdamW | 176.4 | 76.2 | 32.8 / 58.0 | 39.2 / 92.1 | 2.5 / 31.4 | 0.0 |
| FORGE (bf16) | 152.1 | 99.5 | 34.9 / 54.8 | 64.0 / 92.7 | 1.2 / 7.7 | 16.2 |
| FORGE (int8) | 164.7 | 95.6 | 40.7 / 59.4 | 57.7 / 106.5 | 1.4 / 7.9 | 15.1 |

Host time per phase (ms, under the profiler), the 3 largest groups :

| Method | Forward | Backward | Step |
|---|---|---|---|
| vanilla AdamW | elementwise 13.1, copies, new tensors 6.8, Python, waiting (no op) 6.1 | elementwise 16.5, copies, new tensors 6.3, autograd engine 5.9 | optimizer 6.6, elementwise 0.3, copies, new tensors 0.1 |
| fused AdamW | elementwise 15.5, copies, new tensors 7.5, Python, waiting (no op) 6.8 | elementwise 21.1, copies, new tensors 8.3, autograd engine 7.8 | optimizer 2.3, copies, new tensors 0.2, Python, waiting (no op) 0.1 |
| FORGE (bf16) | elementwise 15.5, copies, new tensors 7.6, Python, waiting (no op) 7.2 | elementwise 24.2, FORGE update (Python) 14.5, copies, new tensors 10.3 | optimizer 1.1, copies, new tensors 0.1, Python, waiting (no op) 0.0 |
| FORGE (int8) | elementwise 17.9, copies, new tensors 8.6, Python, waiting (no op) 8.3 | elementwise 20.9, FORGE update (Python) 14.4, copies, new tensors 9.1 | optimizer 1.6, copies, new tensors 0.1, Python, waiting (no op) 0.1 |

![Llama-3.2-1B profile timeline](results/adamw/llama/profile_llama_timeline.png)
![Llama-3.2-1B profile host](results/adamw/llama/profile_llama_host.png)
<!-- /auto -->

### Qwen3-0.6B

<!-- auto:profile adamw/qwen -->
#### Qwen3-0.6B (adamw) : where the time of one step goes (ms, median of 20 steps)

| | vanilla AdamW | fused AdamW | FORGE (bf16) | FORGE (int8) |
|---|:--:|:--:|:--:|:--:|
| **Sync after each phase** | | | | |
| Forward | 98.7 | 121.5 | 123.1 | 123.0 |
| Backward : Python returns | 114.1 | 107.9 | 158.7 | 134.9 |
| Backward : TPU finishes | 178.8 | 221.3 | 238.6 | 252.2 |
| Optimizer step | 44.8 | 26.2 | 9.5 | 9.0 |
| Total | 322.3 | 369.0 | 371.2 | 384.3 |
| **As the benchmark (one sync at the end)** | | | | |
| Python done issuing | 227.3 | 199.5 | 255.4 | 215.7 |
| Step time (TPU done) | **374.5** | **356.5** | **361.6** | **381.9** |
| of which FORGE Python | - | - | 28.5 | 25.7 |
| FORGE updates per step | - | - | 196 | 196 |

#### Qwen3-0.6B (adamw) : profile in detail (median of 20 steps, after 10 warmup steps)

| Method | Step as benchmark (ms) | Python done issuing (ms) | Forward : Python / TPU done | Backward : Python / TPU done | Step : Python / TPU done | FORGE Python (ms) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla AdamW | 374.5 | 227.3 | 72.8 / 98.7 | 114.1 / 178.8 | 27.8 / 44.8 | 0.0 |
| fused AdamW | 356.5 | 199.5 | 74.0 / 121.5 | 107.9 / 221.3 | 6.1 / 26.2 | 0.0 |
| FORGE (bf16) | 361.6 | 255.4 | 99.8 / 123.1 | 158.7 / 238.6 | 3.0 / 9.5 | 28.5 |
| FORGE (int8) | 381.9 | 215.7 | 77.2 / 123.0 | 134.9 / 252.2 | 3.2 / 9.0 | 25.7 |

Host time per phase (ms, under the profiler), the 3 largest groups :

| Method | Forward | Backward | Step |
|---|---|---|---|
| vanilla AdamW | elementwise 34.7, copies, new tensors 19.2, Python, waiting (no op) 16.0 | elementwise 79.7, copies, new tensors 28.5, autograd engine 20.2 | optimizer 23.9, elementwise 0.9, copies, new tensors 0.4 |
| fused AdamW | elementwise 39.5, copies, new tensors 20.8, Python, waiting (no op) 17.9 | elementwise 75.3, copies, new tensors 26.1, autograd engine 18.1 | optimizer 7.6, copies, new tensors 0.5, Python, waiting (no op) 0.1 |
| FORGE (bf16) | elementwise 48.2, copies, new tensors 25.7, Python, waiting (no op) 24.3 | elementwise 74.5, copies, new tensors 29.1, FORGE update (Python) 26.6 | optimizer 4.1, copies, new tensors 0.2, Python, waiting (no op) 0.1 |
| FORGE (int8) | elementwise 38.5, copies, new tensors 21.0, Python, waiting (no op) 19.4 | elementwise 64.0, FORGE update (Python) 26.3, copies, new tensors 24.7 | optimizer 3.6, copies, new tensors 0.2, Python, waiting (no op) 0.1 |

![Qwen3-0.6B profile timeline](results/adamw/qwen/profile_qwen_timeline.png)
![Qwen3-0.6B profile host](results/adamw/qwen/profile_qwen_host.png)
<!-- /auto -->

## Files

```
src/forge_tpu/
  module.py       FusedLinear (the nn.Linear replacement)
  autograd.py     FusedLinearFunction : forward is F.linear, backward does dx + the fused update
  state.py        FusedOptimizerState : step count, hyperparameters and the state tensors of one layer
  _bridge.py      torch <---> jax/pallas : allocate_state, fused_update, the launch replay
  kernel/         the pallas kernel (fused_grad.py), optimizers (optimizers.py), state formats (codecs.py)
benchmarks/
  mlp.py llama.py qwen.py     the models + the measure / report code (mlp.py has the shared part)
  tests.py profiling.py tiler.py
  methods.py                  the optimizers of the benchmarks and their methods (OPTIM : adamw or sgd)
  readme.py                   writes the numbers and plots of results/ into the READMEs (no torch : run it after new results are unpacked)
  colab_cell_*.py             one Colab cell each (OPTIMS at the top : the optimizers to run)
results/adamw/                the latest AdamW results zip, unpacked (README.md : its tables and plots, written by readme.py)
results/sgd/                  the latest SGD results
```

## Known limits

- `torch.compile` doesnt work. (set_ is not tracable, i found no alternative)
- only AdamW and SGD (no momentum) as of now. AdamW states are bf16 or int8 (thats all v5e supports natively). int8 states need `out_features % 64 == 0`.
- FORGE updates the weight and states of a Linear layer only as of now. update of everything else happens in torch's optimizer.step()
- no gradient accumulation yet (the update happens in every backward).
