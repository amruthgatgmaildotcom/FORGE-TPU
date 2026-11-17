### Qwen3-0.6B (adamw) : where the time of one step goes (ms, median of 20 steps)

| | vanilla AdamW | fused AdamW | FORGE (bf16) | FORGE (int8) |
|---|:--:|:--:|:--:|:--:|
| **Sync after each phase** | | | | |
| Forward | 98.7 | 121.5 | 123.1 | 123.0 |
| Backward : Python returns | 114.1 | 107.9 | 158.7 | 134.9 |
| Backward : TPU finishes | 178.8 | 221.3 | 238.6 | 252.2 |
| Optimizer step | 44.8 | 26.2 | 9.5 | 9.0 |
| Total | 322.3 | 369.0 | 371.2 | 384.3 |
| **As the benchmark (one sync at the end)** | | | | |
| Python done issuing | 227.3 | 199.5 | 255.4 | 215.7 |
| Step time (TPU done) | **374.5** | **356.5** | **361.6** | **381.9** |
| of which FORGE Python | - | - | 28.5 | 25.7 |
| FORGE updates per step | - | - | 196 | 196 |

### Qwen3-0.6B (adamw) : profile in detail (median of 20 steps, after 10 warmup steps)

| Method | Step as benchmark (ms) | Python done issuing (ms) | Forward : Python / TPU done | Backward : Python / TPU done | Step : Python / TPU done | FORGE Python (ms) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla AdamW | 374.5 | 227.3 | 72.8 / 98.7 | 114.1 / 178.8 | 27.8 / 44.8 | 0.0 |
| fused AdamW | 356.5 | 199.5 | 74.0 / 121.5 | 107.9 / 221.3 | 6.1 / 26.2 | 0.0 |
| FORGE (bf16) | 361.6 | 255.4 | 99.8 / 123.1 | 158.7 / 238.6 | 3.0 / 9.5 | 28.5 |
| FORGE (int8) | 381.9 | 215.7 | 77.2 / 123.0 | 134.9 / 252.2 | 3.2 / 9.0 | 25.7 |

Host time per phase (ms, under the profiler), the 3 largest groups :

| Method | Forward | Backward | Step |
|---|---|---|---|
| vanilla AdamW | elementwise 34.7, copies, new tensors 19.2, Python, waiting (no op) 16.0 | elementwise 79.7, copies, new tensors 28.5, autograd engine 20.2 | optimizer 23.9, elementwise 0.9, copies, new tensors 0.4 |
| fused AdamW | elementwise 39.5, copies, new tensors 20.8, Python, waiting (no op) 17.9 | elementwise 75.3, copies, new tensors 26.1, autograd engine 18.1 | optimizer 7.6, copies, new tensors 0.5, Python, waiting (no op) 0.1 |
| FORGE (bf16) | elementwise 48.2, copies, new tensors 25.7, Python, waiting (no op) 24.3 | elementwise 74.5, copies, new tensors 29.1, FORGE update (Python) 26.6 | optimizer 4.1, copies, new tensors 0.2, Python, waiting (no op) 0.1 |
| FORGE (int8) | elementwise 38.5, copies, new tensors 21.0, Python, waiting (no op) 19.4 | elementwise 64.0, FORGE update (Python) 26.3, copies, new tensors 24.7 | optimizer 3.6, copies, new tensors 0.2, Python, waiting (no op) 0.1 |
