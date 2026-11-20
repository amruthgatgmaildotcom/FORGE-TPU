### Llama-3.1-8B (adamw, 8 chips, sequence 512) : where the time of one step goes (ms, median of 20 steps)

| | vanilla AdamW | fused AdamW | FORGE (bf16) | FORGE (int8) |
|---|:--:|:--:|:--:|:--:|
| **Sync after each phase** | | | | |
| Forward | 105.9 | 105.6 | 110.7 | 104.6 |
| Backward : Python returns | 78.3 | 72.0 | 112.0 | 107.1 |
| Backward : TPU finishes | 152.1 | 149.4 | 174.1 | 163.0 |
| Optimizer step | 66.3 | 30.5 | 6.2 | 6.5 |
| Total | 324.4 | 285.5 | 290.9 | 274.1 |
| **As the benchmark (one sync at the end)** | | | | |
| Python done issuing | 171.3 | 147.1 | 231.2 | 244.1 |
| Step time (TPU done) | **322.3** | **285.6** | **272.2** | **277.3** |
| of which FORGE Python | - | - | 74.5 | 76.6 |
| FORGE updates per step | - | - | 224 | 224 |

### Llama-3.1-8B (adamw, 8 chips, sequence 512) : profile in detail (median of 20 steps, after 10 warmup steps)

| Method | Step as benchmark (ms) | Python done issuing (ms) | Forward : Python / TPU done | Backward : Python / TPU done | Step : Python / TPU done | FORGE Python (ms) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla AdamW | 322.3 | 171.3 | 68.6 / 105.9 | 78.3 / 152.1 | 21.8 / 66.3 | 0.0 |
| fused AdamW | 285.6 | 147.1 | 62.2 / 105.6 | 72.0 / 149.4 | 5.2 / 30.5 | 0.0 |
| FORGE (bf16) | 272.2 | 231.2 | 59.7 / 110.7 | 112.0 / 174.1 | 1.7 / 6.2 | 74.5 |
| FORGE (int8) | 277.3 | 244.1 | 67.3 / 104.6 | 107.1 / 163.0 | 1.9 / 6.5 | 76.6 |

Host time per phase (ms, under the profiler), the 3 largest groups :

| Method | Forward | Backward | Step |
|---|---|---|---|
| vanilla AdamW | elementwise 26.7, copies, new tensors 14.4, Python, waiting (no op) 13.3 | elementwise 29.2, copies, new tensors 11.1, autograd engine 11.0 | optimizer 19.9, elementwise 0.5, copies, new tensors 0.4 |
| fused AdamW | elementwise 27.4, copies, new tensors 14.2, Python, waiting (no op) 12.2 | elementwise 37.4, copies, new tensors 13.8, autograd engine 12.6 | optimizer 4.9, copies, new tensors 0.3, Python, waiting (no op) 0.1 |
| FORGE (bf16) | elementwise 21.3, Python, waiting (no op) 13.2, copies, new tensors 12.0 | copies, new tensors 70.9, elementwise 36.4, FORGE update (Python) 26.0 | optimizer 1.3, copies, new tensors 0.1, Python, waiting (no op) 0.0 |
| FORGE (int8) | elementwise 24.7, Python, waiting (no op) 14.6, copies, new tensors 12.8 | copies, new tensors 48.7, elementwise 39.6, FORGE update (Python) 28.6 | optimizer 1.6, copies, new tensors 0.1, Python, waiting (no op) 0.1 |
