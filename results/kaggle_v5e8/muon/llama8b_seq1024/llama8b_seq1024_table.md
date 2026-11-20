Llama-3.1-8B (transformers), 8.03B params (random init), batch 1, sequence 1024, tensor parallel over 8 chips, muon, bf16 weights, memory : the largest of the 8 chips. Step time is the median of 190 steps (± std), after 10 warmup steps.

| Method | Peak (GiB) | Step (ms) | TF/s |
|---|:--:|:--:|:--:|
| Muon | 10.20 | 2456.2 ± 20.1 | 21 ± 0 |
| FORGE (Muon) | 9.25 | 2343.5 ± 18.9 | 22 ± 0 |

| Method | Loss at step 200 (last common step) | max \|loss − Muon\| (steps 1 to 200) |
|---|:--:|:--:|
| Muon | 0.1844 | 0.0000 |
| FORGE (Muon) | 0.1401 | 0.4891 |
