MLP 1.09B params (8 blocks, D=4096, FF=16384), 4096 tokens per step, bf16 weights, single TPU. Step time is the median of 214 steps (± std), after 10 warmup steps.

| Method | Peak (GiB) | Step (ms) | TF/s |
|---|:--:|:--:|:--:|
| vanilla AdamW | 12.69 | 259.6 ± 0.0 | 103 ± 0 |
| fused AdamW | 12.69 | 216.7 ± 0.1 | 124 ± 0 |
| FORGE (bf16) | 10.66 | 195.5 ± 0.1 | 137 ± 0 |
| FORGE (int8) | 8.76 | 195.5 ± 0.1 | 137 ± 0 |

| Method | Final loss | max \|loss − vanilla AdamW\| |
|---|:--:|:--:|
| vanilla AdamW | 0.3941 | 0.0000 |
| fused AdamW | 0.3944 | 0.0005 |
| FORGE (bf16) | 0.3943 | 0.0005 |
| FORGE (int8) | 0.4016 | 0.3211 |
