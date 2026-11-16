Llama-3.2-1B (transformers), 1.24B params (random init), batch 1, sequence 1024, sgd, bf16 weights, single TPU. Step time is the median of 290 steps (± std), after 10 warmup steps.

| Method | Peak (GiB) | Step (ms) | TF/s |
|---|:--:|:--:|:--:|
| vanilla SGD | 8.47 | 172.2 ± 16.9 | 46 ± 5 |
| fused SGD | 6.75 | 162.6 ± 15.4 | 49 ± 5 |
| FORGE (SGD) | 6.71 | 150.3 ± 13.5 | 53 ± 5 |

| Method | Loss at step 300 (last common step) | max \|loss − vanilla SGD\| (steps 1 to 300) |
|---|:--:|:--:|
| vanilla SGD | 5.4212 | 0.0000 |
| fused SGD | 5.8439 | 1.8330 |
| FORGE (SGD) | 5.7465 | 2.4620 |
