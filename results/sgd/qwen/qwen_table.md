Qwen3-0.6B (transformers), 0.60B params (random init), batch 1, sequence 1024, sgd, bf16 weights, single TPU. Step time is the median of 127 steps (± std), after 10 warmup steps.

| Method | Peak (GiB) | Step (ms) | TF/s |
|---|:--:|:--:|:--:|
| vanilla SGD | 5.87 | 361.0 ± 18.9 | 12 ± 1 |
| fused SGD | 5.87 | 349.4 ± 32.0 | 13 ± 1 |
| FORGE (SGD) | 5.86 | 354.9 ± 24.0 | 12 ± 1 |

| Method | Loss at step 131 (last common step) | max \|loss − vanilla SGD\| (steps 1 to 131) |
|---|:--:|:--:|
| vanilla SGD | 9.3569 | 0.0000 |
| fused SGD | 9.3581 | 0.0013 |
| FORGE (SGD) | 9.3575 | 0.0011 |
