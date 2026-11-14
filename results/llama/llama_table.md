Llama-3.2-1B architecture, 1.24B params (random init), batch 1, sequence 1024, bf16 weights, single TPU. Step time is the median of 253 steps (± std), after 10 warmup steps.

| Method | Peak (GiB) | Step (ms) | TF/s |
|---|:--:|:--:|:--:|
| vanilla AdamW | 11.92 | 195.6 ± 10.9 | 41 ± 2 |
| fused AdamW | 11.20 | 133.2 ± 7.1 | 60 ± 3 |
| FORGE (bf16) | 11.12 | 125.8 ± 2.3 | 64 ± 1 |
| FORGE (int8) | 9.38 | 119.0 ± 8.1 | 67 ± 4 |

| Method | Final loss | max \|loss − vanilla AdamW\| |
|---|:--:|:--:|
| vanilla AdamW | 0.0619 | 0.0000 |
| fused AdamW | 0.0508 | 8.3015 |
| FORGE (bf16) | 0.0237 | 4.2468 |
| FORGE (int8) | 0.0206 | 11.6855 |
