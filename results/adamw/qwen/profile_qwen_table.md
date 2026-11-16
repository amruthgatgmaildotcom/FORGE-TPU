### Qwen3-0.6B (adamw) : where the time of one step goes (ms, median of 20 steps)

| | vanilla AdamW | fused AdamW | FORGE (bf16) | FORGE (int8) |
|---|:--:|:--:|:--:|:--:|
| **Sync after each phase** | | | | |
| Forward | 127.8 | 119.3 | 120.8 | 132.8 |
| Backward : Python returns | 110.8 | 124.8 | 148.5 | 138.6 |
| Backward : TPU finishes | 229.2 | 217.3 | 238.7 | 256.5 |
| Optimizer step | 45.5 | 25.3 | 9.0 | 8.4 |
| Total | 402.5 | 361.9 | 368.5 | 397.6 |
| **As the benchmark (one sync at the end)** | | | | |
| Python done issuing | 227.1 | 181.2 | 223.1 | 246.3 |
| Step time (TPU done) | **434.3** | **355.1** | **364.9** | **357.5** |
| of which FORGE Python | - | - | 28.8 | 34.4 |
| FORGE updates per step | - | - | 196 | 196 |

### Qwen3-0.6B (adamw) : profile in detail (median of 20 steps, after 10 warmup steps)

| Method | Step as benchmark (ms) | Python done issuing (ms) | Forward : Python / TPU done | Backward : Python / TPU done | Step : Python / TPU done | FORGE Python (ms) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla AdamW | 434.3 | 227.1 | 62.7 / 127.8 | 110.8 / 229.2 | 32.7 / 45.5 | 0.0 |
| fused AdamW | 355.1 | 181.2 | 72.4 / 119.3 | 124.8 / 217.3 | 5.5 / 25.3 | 0.0 |
| FORGE (bf16) | 364.9 | 223.1 | 74.7 / 120.8 | 148.5 / 238.7 | 3.4 / 9.0 | 28.8 |
| FORGE (int8) | 357.5 | 246.3 | 69.4 / 132.8 | 138.6 / 256.5 | 3.1 / 8.4 | 34.4 |

Host time per phase (ms, under the profiler), the 3 largest groups :

| Method | Forward | Backward | Step |
|---|---|---|---|
| vanilla AdamW | elementwise 29.0, copies, new tensors 15.8, Python, waiting (no op) 13.9 | elementwise 85.1, copies, new tensors 32.1, autograd engine 22.7 | optimizer 29.2, elementwise 0.8, copies, new tensors 0.3 |
| fused AdamW | elementwise 33.3, copies, new tensors 18.1, Python, waiting (no op) 15.9 | elementwise 73.9, copies, new tensors 27.8, autograd engine 20.8 | optimizer 5.4, copies, new tensors 0.3, Python, waiting (no op) 0.1 |
| FORGE (bf16) | elementwise 38.8, copies, new tensors 20.7, Python, waiting (no op) 19.6 | elementwise 67.3, FORGE update (Python) 27.4, copies, new tensors 25.9 | optimizer 3.6, copies, new tensors 0.2, Python, waiting (no op) 0.1 |
| FORGE (int8) | elementwise 32.8, copies, new tensors 17.2, Python, waiting (no op) 16.4 | elementwise 62.0, FORGE update (Python) 26.1, copies, new tensors 23.6 | optimizer 2.8, copies, new tensors 0.1, Python, waiting (no op) 0.1 |
