Llama-3.2-1B (transformers), 1.24B params (random init), batch 1, sequence 1024, bf16 weights, single TPU. Step time is the median of 201 steps (± std), after 10 warmup steps.

| Method | Peak (GiB) | Step (ms) | TF/s |
|---|:--:|:--:|:--:|
| vanilla AdamW | 11.75 | 237.0 ± 18.2 | 34 ± 2 |
| fused AdamW | 11.37 | 173.1 ± 16.7 | 46 ± 4 |
| FORGE (bf16) | 11.31 | 170.6 ± 10.1 | 47 ± 3 |
| FORGE (int8) | 9.56 | 179.7 ± 8.4 | 45 ± 2 |

| Method | Final loss | max \|loss − vanilla AdamW\| |
|---|:--:|:--:|
| vanilla AdamW | 0.1046 | 0.0000 |
| fused AdamW | 0.0315 | 8.9980 |
| FORGE (bf16) | 0.0230 | 10.5183 |
| FORGE (int8) | 0.1155 | 6.0822 |
