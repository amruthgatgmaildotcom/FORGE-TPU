Llama-3.2-1B (transformers), 1.24B params (random init), batch 1, sequence 1024, sgd, bf16 weights, single TPU. Step time is the median of 290 steps (± std), after 10 warmup steps.

| Method | Peak (GiB) | Step (ms) | TF/s |
|---|:--:|:--:|:--:|
| vanilla SGD | 8.49 | 169.1 ± 16.1 | 47 ± 5 |
| fused SGD | 6.75 | 157.9 ± 13.4 | 51 ± 5 |
| FORGE (SGD) | 6.71 | 153.3 ± 7.5 | 52 ± 2 |

| Method | Loss at step 300 (last common step) | max \|loss − vanilla SGD\| (steps 1 to 300) |
|---|:--:|:--:|
| vanilla SGD | 5.4212 | 0.0000 |
| fused SGD | 5.8439 | 1.8330 |
| FORGE (SGD) | 5.7465 | 2.4620 |
