Llama-3.2-1B (transformers), 1.24B params (random init), batch 1, sequence 1024, adamw, bf16 weights, single TPU. Step time is the median of 214 steps (± std), after 10 warmup steps.

| Method | Peak (GiB) | Step (ms) | TF/s |
|---|:--:|:--:|:--:|
| vanilla AdamW | 12.04 | 221.9 ± 12.3 | 36 ± 2 |
| fused AdamW | 11.37 | 176.0 ± 9.3 | 45 ± 2 |
| FORGE (bf16) | 11.31 | 169.3 ± 20.4 | 47 ± 6 |
| FORGE (int8) | 9.57 | 160.5 ± 18.3 | 50 ± 6 |

| Method | Loss at step 220 (last common step) | max \|loss − vanilla AdamW\| (steps 1 to 220) |
|---|:--:|:--:|
| vanilla AdamW | 0.0442 | 0.0000 |
| fused AdamW | 0.0428 | 8.9854 |
| FORGE (bf16) | 0.0355 | 10.5057 |
| FORGE (int8) | 0.1688 | 7.5098 |
