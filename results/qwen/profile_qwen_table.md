### Qwen3-0.6B : where the time of one step goes (ms, median of 20 steps)

| | vanilla AdamW | fused AdamW | FORGE (bf16) | FORGE (int8) |
|---|:--:|:--:|:--:|:--:|
| **Sync after each phase** | | | | |
| Forward | 124.6 | 121.3 | 129.0 | 130.4 |
| Backward : Python returns | 132.5 | 96.3 | 150.8 | 141.3 |
| Backward : TPU finishes | 221.2 | 226.8 | 251.1 | 259.2 |
| Optimizer step | 157.9 | 25.8 | 10.4 | 9.2 |
| Total | 503.6 | 373.8 | 390.5 | 398.8 |
| **As the benchmark (one sync at the end)** | | | | |
| Python done issuing | 266.9 | 175.0 | 198.6 | 282.5 |
| Step time (TPU done) | **504.9** | **377.6** | **379.6** | **399.0** |
| of which FORGE Python | - | - | 28.9 | 39.8 |
| FORGE updates per step | - | - | 196 | 196 |

### Qwen3-0.6B : profile in detail (median of 20 steps, after 10 warmup steps)

| Method | Step as benchmark (ms) | Python done issuing (ms) | Forward : Python / TPU done | Backward : Python / TPU done | Step : Python / TPU done | FORGE Python (ms) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla AdamW | 504.9 | 266.9 | 60.8 / 124.6 | 132.5 / 221.2 | 57.5 / 157.9 | 0.0 |
| fused AdamW | 377.6 | 175.0 | 80.1 / 121.3 | 96.3 / 226.8 | 5.6 / 25.8 | 0.0 |
| FORGE (bf16) | 379.6 | 198.6 | 68.0 / 129.0 | 150.8 / 251.1 | 3.4 / 10.4 | 28.9 |
| FORGE (int8) | 399.0 | 282.5 | 80.6 / 130.4 | 141.3 / 259.2 | 3.2 / 9.2 | 39.8 |

Host time per phase (ms, under the profiler), the 3 largest groups :

| Method | Forward | Backward | Step |
|---|---|---|---|
| vanilla AdamW | elementwise 33.0, copies, new tensors 18.8, Python, waiting (no op) 15.9 | elementwise 69.2, copies, new tensors 26.2, autograd engine 18.9 | optimizer 30.3, elementwise 27.4, copies, new tensors 4.7 |
| fused AdamW | elementwise 40.6, copies, new tensors 22.0, Python, waiting (no op) 18.5 | elementwise 61.3, copies, new tensors 22.7, autograd engine 16.6 | optimizer 7.2, copies, new tensors 0.4, Python, waiting (no op) 0.1 |
| FORGE (bf16) | elementwise 30.3, copies, new tensors 16.5, Python, waiting (no op) 15.9 | elementwise 59.3, FORGE update (Python) 26.4, copies, new tensors 22.8 | optimizer 4.2, copies, new tensors 0.2, Python, waiting (no op) 0.1 |
| FORGE (int8) | elementwise 40.9, copies, new tensors 22.6, Python, waiting (no op) 20.5 | elementwise 67.9, FORGE update (Python) 29.6, copies, new tensors 26.6 | optimizer 3.6, copies, new tensors 0.2, Python, waiting (no op) 0.1 |
