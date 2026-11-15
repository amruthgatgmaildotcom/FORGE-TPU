Qwen3-0.6B (transformers), 0.60B params (random init), batch 1, sequence 1024, bf16 weights, single TPU. Step time is the median of 81 steps (± std), after 10 warmup steps.

| Method | Peak (GiB) | Step (ms) | TF/s |
|---|:--:|:--:|:--:|
| vanilla AdamW | 8.09 | 520.5 ± 50.5 | 8 ± 1 |
| fused AdamW | 8.11 | 353.7 ± 56.1 | 12 ± 2 |
| FORGE (bf16) | 8.10 | 381.4 ± 56.2 | 11 ± 2 |
| FORGE (int8) | 7.33 | 431.7 ± 33.7 | 10 ± 1 |

| Method | Final loss | max \|loss − vanilla AdamW\| |
|---|:--:|:--:|
| vanilla AdamW | 6.1731 | 0.0000 |
| fused AdamW | 6.1243 | 0.0132 |
| FORGE (bf16) | 6.1373 | 0.0136 |
| FORGE (int8) | 6.2357 | 0.9443 |
