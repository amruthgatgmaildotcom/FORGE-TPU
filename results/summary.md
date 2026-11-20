# Results

Written by `benchmarks/report.py` from the files in `results/`. One TPU v5e chip (197 TF/s bf16). Mean over the seeds; ± : half the gap between the seeds (one seed : ± the std of the timed steps). MFU = TF/s / 197.

## MLP-1.1B · adamw

1.09B params, 4096 tokens per step, lr 0.0003, 2 seed(s), 20 warmup steps (not timed), up to 300 steps or 120 s.

| Method | Peak (GiB) | Step (ms) | TF/s | MFU | Peak vs. vanilla | Step vs. vanilla | Peak vs. best torch | Step vs. best torch | Final loss |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla AdamW | 13.91 | 259.5 ± 0.0 | 103.3 | 52% | — | — | +9.6% | 0.84× | 0.3959 |
| fused AdamW | 12.69 | 216.7 ± 0.1 | 123.7 | 63% | -8.8% | 1.20× | — | — | 0.3960 |
| FORGE (bf16) | 10.67 | 187.5 ± 0.0 | 142.9 | 73% | -23.3% | 1.38× | -16.0% | 1.16× | 0.3960 |
| FORGE (int8) | 8.76 | 188.7 ± 0.0 | 142.0 | 72% | -37.0% | 1.38× | -31.0% | 1.15× | 0.4016 |

Every seed :

| Method | Seed | Steps | Peak (GiB) | weights | states | grads | activations | Step (ms) | TF/s | Final loss |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla AdamW | 0 | 300 | 13.913 | 2.031 | 4.063 | 2.031 | 5.788 | 259.5 ± 0.1 | 103.3 | 0.3959 |
| vanilla AdamW | 1 | 300 | 13.913 | 2.031 | 4.063 | 2.031 | 5.788 | 259.5 ± 0.1 | 103.3 | 0.3958 |
| fused AdamW | 0 | 300 | 12.692 | 2.031 | 4.063 | 2.031 | 4.567 | 216.8 ± 0.1 | 123.6 | 0.3960 |
| fused AdamW | 1 | 300 | 12.692 | 2.031 | 4.063 | 2.031 | 4.567 | 216.7 ± 0.1 | 123.7 | 0.3960 |
| FORGE (bf16) | 0 | 300 | 10.666 | 2.031 | 4.062 | 0.000 | 4.572 | 187.5 ± 0.1 | 142.9 | 0.3961 |
| FORGE (bf16) | 1 | 300 | 10.666 | 2.031 | 4.062 | 0.000 | 4.572 | 187.5 ± 0.1 | 142.9 | 0.3960 |
| FORGE (int8) | 0 | 300 | 8.763 | 2.031 | 2.158 | 0.000 | 4.573 | 188.8 ± 0.1 | 142.0 | 0.4016 |
| FORGE (int8) | 1 | 300 | 8.763 | 2.031 | 2.158 | 0.000 | 4.573 | 188.7 ± 0.1 | 142.0 | 0.4015 |

## Llama-3.2-1B · adamw

1.24B params, 512 tokens per step, lr 0.0001, 2 seed(s), 20 warmup steps (not timed), up to 300 steps or 120 s.

| Method | Peak (GiB) | Step (ms) | TF/s | MFU | Peak vs. vanilla | Step vs. vanilla | Peak vs. best torch | Step vs. best torch | Final loss |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla AdamW | 11.79 | 213.7 ± 3.0 | 18.3 | 9% | — | — | +15.0% | 0.76× | 0.0123 |
| fused AdamW | 10.26 | 161.4 ± 7.3 | 24.2 | 12% | -13.0% | 1.32× | — | — | 0.0121 |
| FORGE (bf16) | 9.35 | 152.1 ± 2.9 | 25.7 | 13% | -20.7% | 1.41× | -8.8% | 1.06× | 0.0121 |
| FORGE (int8) | 7.65 | 162.7 ± 0.4 | 24.0 | 12% | -35.2% | 1.31× | -25.5% | 0.99× | 0.0330 |

Every seed :

| Method | Seed | Steps | Peak (GiB) | weights | states | grads | activations | Step (ms) | TF/s | Final loss |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla AdamW | 0 | 300 | 11.795 | 2.302 | 4.604 | 2.302 | 2.587 | 216.7 ± 8.7 | 18.0 | 0.0123 |
| vanilla AdamW | 1 | 300 | 11.795 | 2.302 | 4.604 | 2.302 | 2.587 | 210.7 ± 16.8 | 18.5 | 0.0123 |
| fused AdamW | 0 | 300 | 10.266 | 2.302 | 4.604 | 2.302 | 1.059 | 154.1 ± 15.6 | 25.3 | 0.0121 |
| fused AdamW | 1 | 300 | 10.250 | 2.302 | 4.604 | 2.302 | 1.043 | 168.6 ± 4.4 | 23.1 | 0.0121 |
| FORGE (bf16) | 0 | 300 | 9.353 | 2.302 | 4.604 | 0.489 | 1.958 | 149.2 ± 21.2 | 26.1 | 0.0121 |
| FORGE (bf16) | 1 | 300 | 9.353 | 2.302 | 4.604 | 0.489 | 1.958 | 155.0 ± 8.2 | 25.2 | 0.0121 |
| FORGE (int8) | 0 | 300 | 7.645 | 2.302 | 2.905 | 0.489 | 1.950 | 162.2 ± 11.5 | 24.0 | 0.0310 |
| FORGE (int8) | 1 | 300 | 7.645 | 2.302 | 2.905 | 0.489 | 1.950 | 163.1 ± 26.8 | 23.9 | 0.0351 |

Where the time of one step goes (`profiling.py`, median of 20 steps, ms) :

| Method | Forward | Backward | Optimizer step | Sum (sync per phase) | Step (one sync) | Python done issuing | FORGE Python | FORGE updates |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla AdamW | 52.3 | 83.1 | 79.8 | 215.2 | 188.7 | 90.3 | 0.0 | — |
| fused AdamW | 53.6 | 85.3 | 33.0 | 171.8 | 164.5 | 86.8 | 0.0 | — |
| FORGE (bf16) | 54.7 | 94.3 | 8.4 | 157.3 | 150.3 | 102.2 | 14.5 | 112 |
| FORGE (int8) | 53.4 | 99.3 | 8.1 | 160.9 | 158.6 | 102.5 | 15.8 | 112 |

## Qwen3-0.6B · adamw

0.60B params, 512 tokens per step, lr 0.0001, 2 seed(s), 20 warmup steps (not timed), up to 300 steps or 120 s.

| Method | Peak (GiB) | Step (ms) | TF/s | MFU | Peak vs. vanilla | Step vs. vanilla | Peak vs. best torch | Step vs. best torch | Final loss |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla AdamW | 6.03 | 377.6 ± 3.8 | 5.3 | 3% | — | — | +2.2% | 0.94× | 6.0810 |
| fused AdamW | 5.90 | 355.7 ± 5.3 | 5.7 | 3% | -2.2% | 1.06× | — | — | 5.9387 |
| FORGE (bf16) | 5.87 | 367.4 ± 2.2 | 5.5 | 3% | -2.7% | 1.03× | -0.5% | 0.97× | 6.0441 |
| FORGE (int8) | 5.10 | 364.1 ± 1.6 | 5.5 | 3% | -15.4% | 1.04× | -13.5% | 0.98× | 6.0675 |

Every seed :

| Method | Seed | Steps | Peak (GiB) | weights | states | grads | activations | Step (ms) | TF/s | Final loss |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla AdamW | 0 | 225 | 5.923 | 1.110 | 2.220 | 1.110 | 1.482 | 381.4 ± 32.2 | 5.3 | 6.0942 |
| vanilla AdamW | 1 | 238 | 6.133 | 1.110 | 2.220 | 1.110 | 1.692 | 373.9 ± 40.7 | 5.4 | 6.0595 |
| fused AdamW | 0 | 228 | 5.895 | 1.110 | 2.220 | 1.110 | 1.455 | 361.0 ± 32.7 | 5.6 | 6.0968 |
| fused AdamW | 1 | 241 | 5.895 | 1.110 | 2.220 | 1.110 | 1.455 | 350.4 ± 41.5 | 5.7 | 5.7541 |
| FORGE (bf16) | 0 | 222 | 5.868 | 1.110 | 2.220 | 0.290 | 2.247 | 365.3 ± 40.2 | 5.5 | 6.0904 |
| FORGE (bf16) | 1 | 221 | 5.868 | 1.110 | 2.220 | 0.290 | 2.247 | 369.6 ± 34.7 | 5.4 | 5.9980 |
| FORGE (int8) | 0 | 220 | 5.102 | 1.110 | 1.451 | 0.290 | 2.250 | 362.5 ± 43.0 | 5.5 | 6.0987 |
| FORGE (int8) | 1 | 217 | 5.102 | 1.110 | 1.451 | 0.290 | 2.250 | 365.8 ± 41.9 | 5.5 | 6.0315 |

Where the time of one step goes (`profiling.py`, median of 20 steps, ms) :

| Method | Forward | Backward | Optimizer step | Sum (sync per phase) | Step (one sync) | Python done issuing | FORGE Python | FORGE updates |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla AdamW | 120.4 | 220.1 | 46.1 | 386.6 | 376.8 | 222.9 | 0.0 | — |
| fused AdamW | 112.7 | 203.5 | 26.0 | 342.2 | 351.1 | 181.3 | 0.0 | — |
| FORGE (bf16) | 131.7 | 247.5 | 9.5 | 388.7 | 356.0 | 222.4 | 24.2 | 196 |
| FORGE (int8) | 116.4 | 231.4 | 8.7 | 356.4 | 335.3 | 231.6 | 27.2 | 196 |

## Correctness · adamw

FORGE correctness (adamw) : 153 / 160 checks pass ([tests.md](adamw/tests.md)). Fails : step adamw int8 state, bfloat16 w, 1500x1536x2048 : w(ulp); step adamw int8 state, bfloat16 w, 256x1024x512, g x1e-06 : w(ulp); step adamw int8 state, bfloat16 w, 256x1024x512, g x1e-06 : v; step adamw bf16 state, bfloat16 w, 64x512x256, bias : db; chain adamw int8 state, 2 layers, 10 steps (worst) : l2 w(ulp); chain adamw int8 state, 2 layers, 10 steps (worst) : l1 w(ulp); model forge_int8 vs PyTorch bf16 adamw : max |loss - ref|

## Llama-3.2-1B · muon

1.24B params, 512 tokens per step, lr 0.0001, 1 seed(s), 20 warmup steps (not timed), up to 300 steps or 180 s.

| Method | Peak (GiB) | Step (ms) | TF/s | MFU | Peak vs. vanilla | Step vs. vanilla | Peak vs. best torch | Step vs. best torch | Final loss |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Muon | 9.26 | 910.6 ± 31.6 | 4.3 | 2% | — | — | — | — | 0.1838 |
| FORGE (Muon) | 7.54 | 827.5 ± 29.3 | 4.7 | 2% | -18.6% | 1.10× | -18.6% | 1.10× | 0.1906 |

Every seed :

| Method | Seed | Steps | Peak (GiB) | weights | states | grads | activations | Step (ms) | TF/s | Final loss |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Muon | 0 | 184 | 9.263 | 2.302 | 2.791 | 2.302 | 1.868 | 910.6 ± 31.6 | 4.3 | 0.1838 |
| FORGE (Muon) | 0 | 201 | 7.539 | 2.302 | 2.791 | 0.489 | 1.957 | 827.5 ± 29.3 | 4.7 | 0.1906 |

## Qwen3-0.6B · muon

0.60B params, 512 tokens per step, lr 0.0001, 1 seed(s), 20 warmup steps (not timed), up to 300 steps or 180 s.

| Method | Peak (GiB) | Step (ms) | TF/s | MFU | Peak vs. vanilla | Step vs. vanilla | Peak vs. best torch | Step vs. best torch | Final loss |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Muon | 5.06 | 1057.6 ± 87.1 | 1.9 | 1% | — | — | — | — | 0.4212 |
| FORGE (Muon) | 5.04 | 1006.9 ± 64.9 | 2.0 | 1% | -0.2% | 1.05× | -0.2% | 1.05× | 0.4470 |

Every seed :

| Method | Seed | Steps | Peak (GiB) | weights | states | grads | activations | Step (ms) | TF/s | Final loss |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Muon | 0 | 157 | 5.056 | 1.110 | 1.400 | 1.110 | 1.435 | 1057.6 ± 87.1 | 1.9 | 0.4212 |
| FORGE (Muon) | 0 | 164 | 5.044 | 1.110 | 1.400 | 0.290 | 2.244 | 1006.9 ± 64.9 | 2.0 | 0.4470 |
