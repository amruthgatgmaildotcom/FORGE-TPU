Qwen3-0.6B (transformers), 0.60B params (random init), batch 1, sequence 1024, sgd, bf16 weights, single TPU. Step time is the median of 990 steps (± std), after 10 warmup steps.

| Method | Peak (GiB) | Step (ms) | TF/s |
|---|:--:|:--:|:--:|
| vanilla SGD | 5.87 | 355.2 ± 44.2 | 12 ± 2 |
| fused SGD | 5.87 | 351.6 ± 38.4 | 12 ± 2 |
| FORGE (SGD) | 5.86 | 350.0 ± 36.9 | 13 ± 1 |

| Method | Loss at step 1000 (last common step) | max \|loss − vanilla SGD\| (steps 1 to 1000) |
|---|:--:|:--:|
| vanilla SGD | 6.5139 | 0.0000 |
| fused SGD | 6.5139 | 0.0020 |
| FORGE (SGD) | 6.5136 | 0.0017 |
