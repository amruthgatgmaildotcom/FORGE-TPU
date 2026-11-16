MLP 1.09B params (8 blocks, D=4096, FF=16384), 4096 tokens per step, adamw, bf16 weights, single TPU. Step time is the median of 214 steps (± std), after 10 warmup steps.

| Method | Peak (GiB) | Step (ms) | TF/s |
|---|:--:|:--:|:--:|
| vanilla AdamW | 13.91 | 259.5 ± 0.1 | 103 ± 0 |
| fused AdamW | 12.69 | 216.7 ± 0.1 | 124 ± 0 |
| FORGE (bf16) | 10.67 | 187.9 ± 0.1 | 143 ± 0 |
| FORGE (int8) | 8.76 | 188.9 ± 0.1 | 142 ± 0 |

| Method | Loss at step 224 (last common step) | max \|loss − vanilla AdamW\| (steps 1 to 224) |
|---|:--:|:--:|
| vanilla AdamW | 0.3941 | 0.0000 |
| fused AdamW | 0.3941 | 0.0004 |
| FORGE (bf16) | 0.3941 | 0.0004 |
| FORGE (int8) | 0.4027 | 0.3211 |
