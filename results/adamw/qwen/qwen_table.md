Qwen3-0.6B (transformers), 0.60B params (random init), batch 1, sequence 1024, adamw, bf16 weights, single TPU. Step time is the median of 107 steps (± std), after 10 warmup steps.

| Method | Peak (GiB) | Step (ms) | TF/s |
|---|:--:|:--:|:--:|
| vanilla AdamW | 8.14 | 393.3 ± 43.2 | 11 ± 1 |
| fused AdamW | 8.11 | 357.4 ± 17.0 | 12 ± 1 |
| FORGE (bf16) | 8.10 | 305.0 ± 43.6 | 14 ± 2 |
| FORGE (int8) | 7.33 | 362.7 ± 28.3 | 12 ± 1 |

| Method | Loss at step 99 (last common step) | max \|loss − vanilla AdamW\| (steps 1 to 99) |
|---|:--:|:--:|
| vanilla AdamW | 6.1558 | 0.0000 |
| fused AdamW | 6.1519 | 0.0127 |
| FORGE (bf16) | 6.1519 | 0.0134 |
| FORGE (int8) | 6.1603 | 0.9434 |
