### Llama-3.2-1B (adamw) : where the time of one step goes (ms, median of 20 steps)

| | vanilla AdamW | fused AdamW | FORGE (bf16) | FORGE (int8) |
|---|:--:|:--:|:--:|:--:|
| **Sync after each phase** | | | | |
| Forward | 61.4 | 58.0 | 54.8 | 59.4 |
| Backward : Python returns | 36.7 | 39.2 | 64.0 | 57.7 |
| Backward : TPU finishes | 92.7 | 92.1 | 92.7 | 106.5 |
| Optimizer step | 79.2 | 31.4 | 7.7 | 7.9 |
| Total | 233.3 | 181.5 | 155.1 | 173.8 |
| **As the benchmark (one sync at the end)** | | | | |
| Python done issuing | 70.2 | 76.2 | 99.5 | 95.6 |
| Step time (TPU done) | **195.1** | **176.4** | **152.1** | **164.7** |
| of which FORGE Python | - | - | 16.2 | 15.1 |
| FORGE updates per step | - | - | 112 | 112 |

### Llama-3.2-1B (adamw) : profile in detail (median of 20 steps, after 10 warmup steps)

| Method | Step as benchmark (ms) | Python done issuing (ms) | Forward : Python / TPU done | Backward : Python / TPU done | Step : Python / TPU done | FORGE Python (ms) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla AdamW | 195.1 | 70.2 | 28.2 / 61.4 | 36.7 / 92.7 | 9.5 / 79.2 | 0.0 |
| fused AdamW | 176.4 | 76.2 | 32.8 / 58.0 | 39.2 / 92.1 | 2.5 / 31.4 | 0.0 |
| FORGE (bf16) | 152.1 | 99.5 | 34.9 / 54.8 | 64.0 / 92.7 | 1.2 / 7.7 | 16.2 |
| FORGE (int8) | 164.7 | 95.6 | 40.7 / 59.4 | 57.7 / 106.5 | 1.4 / 7.9 | 15.1 |

Host time per phase (ms, under the profiler), the 3 largest groups :

| Method | Forward | Backward | Step |
|---|---|---|---|
| vanilla AdamW | elementwise 13.1, copies, new tensors 6.8, Python, waiting (no op) 6.1 | elementwise 16.5, copies, new tensors 6.3, autograd engine 5.9 | optimizer 6.6, elementwise 0.3, copies, new tensors 0.1 |
| fused AdamW | elementwise 15.5, copies, new tensors 7.5, Python, waiting (no op) 6.8 | elementwise 21.1, copies, new tensors 8.3, autograd engine 7.8 | optimizer 2.3, copies, new tensors 0.2, Python, waiting (no op) 0.1 |
| FORGE (bf16) | elementwise 15.5, copies, new tensors 7.6, Python, waiting (no op) 7.2 | elementwise 24.2, FORGE update (Python) 14.5, copies, new tensors 10.3 | optimizer 1.1, copies, new tensors 0.1, Python, waiting (no op) 0.0 |
| FORGE (int8) | elementwise 17.9, copies, new tensors 8.6, Python, waiting (no op) 8.3 | elementwise 20.9, FORGE update (Python) 14.4, copies, new tensors 9.1 | optimizer 1.6, copies, new tensors 0.1, Python, waiting (no op) 0.1 |
