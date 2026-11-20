Llama-3.1-8B (transformers), 8.03B params (random init), batch 1, sequence 512, tensor parallel over 8 chips, adamw, bf16 weights, memory : the largest of the 8 chips. Step time is the median of 190 steps (± std), after 10 warmup steps.

| Method | Peak (GiB) | Step (ms) | TF/s |
|---|:--:|:--:|:--:|
| vanilla AdamW | 9.43 | 309.5 ± 18.5 | 81 ± 4 |
| fused AdamW | 8.08 | 280.4 ± 12.7 | 89 ± 4 |
| FORGE (bf16) | 7.29 | 273.5 ± 13.7 | 92 ± 4 |
| FORGE (int8) | 5.78 | 272.2 ± 15.1 | 92 ± 5 |

| Method | Loss at step 200 (last common step) | max \|loss − vanilla AdamW\| (steps 1 to 200) |
|---|:--:|:--:|
| vanilla AdamW | 0.0024 | 0.0000 |
| fused AdamW | 0.0025 | 0.1466 |
| FORGE (bf16) | 0.0025 | 0.0916 |
| FORGE (int8) | 0.0059 | 13.3512 |
