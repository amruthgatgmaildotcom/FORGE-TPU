Qwen3-0.6B architecture, 0.60B params (random init), batch 1, sequence 1024, bf16 weights, single TPU. Step time is the median of 146 steps (± std), after 10 warmup steps.

| Method | Peak (GiB) | Step (ms) | TF/s |
|---|:--:|:--:|:--:|
| vanilla AdamW | 7.83 | 323.0 ± 27.0 | 14 ± 1 |
| fused AdamW | 7.86 | 186.7 ± 8.2 | 23 ± 1 |
| FORGE (bf16) | 7.82 | 189.7 ± 18.1 | 23 ± 2 |
| FORGE (int8) | 7.06 | 189.4 ± 24.6 | 23 ± 3 |

| Method | Final loss | max \|loss − vanilla AdamW\| |
|---|:--:|:--:|
| vanilla AdamW | 6.1134 | 0.0000 |
| fused AdamW | 6.0749 | 0.0147 |
| FORGE (bf16) | 6.0553 | 0.0143 |
| FORGE (int8) | 6.0279 | 0.0958 |
