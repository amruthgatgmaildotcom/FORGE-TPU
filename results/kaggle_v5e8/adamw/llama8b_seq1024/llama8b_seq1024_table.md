Llama-3.1-8B (transformers), 8.03B params (random init), batch 1, sequence 1024, tensor parallel over 8 chips, adamw, bf16 weights, memory : the largest of the 8 chips. Step time is the median of 190 steps (± std), after 10 warmup steps.

| Method | Peak (GiB) | Step (ms) | TF/s |
|---|:--:|:--:|:--:|
| vanilla AdamW | 9.65 | 324.0 ± 11.0 | 157 ± 5 |
| fused AdamW | 8.94 | 285.8 ± 12.8 | 178 ± 8 |
| FORGE (bf16) | 8.84 | 272.8 ± 10.9 | 187 ± 7 |
| FORGE (int8) | 7.33 | 279.9 ± 13.4 | 182 ± 8 |

| Method | Loss at step 200 (last common step) | max \|loss − vanilla AdamW\| (steps 1 to 200) |
|---|:--:|:--:|
| vanilla AdamW | 0.0017 | 0.0000 |
| fused AdamW | 0.0058 | 15.5372 |
| FORGE (bf16) | 0.0036 | 15.9992 |
| FORGE (int8) | 0.0072 | 18.4168 |
