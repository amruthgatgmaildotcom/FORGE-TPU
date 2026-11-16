MLP 1.09B params (8 blocks, D=4096, FF=16384), 4096 tokens per step, sgd, bf16 weights, single TPU. Step time is the median of 265 steps (± std), after 10 warmup steps.

| Method | Peak (GiB) | Step (ms) | TF/s |
|---|:--:|:--:|:--:|
| vanilla SGD | 10.53 | 213.2 ± 0.1 | 126 ± 0 |
| fused SGD | 8.63 | 203.3 ± 0.1 | 132 ± 0 |
| FORGE (SGD) | 6.45 | 177.8 ± 0.1 | 151 ± 0 |

| Method | Loss at step 275 (last common step) | max \|loss − vanilla SGD\| (steps 1 to 275) |
|---|:--:|:--:|
| vanilla SGD | 0.7278 | 0.0000 |
| fused SGD | 0.7278 | 0.0000 |
| FORGE (SGD) | 0.7278 | 0.0000 |
