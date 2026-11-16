### Llama-3.2-1B (adamw) : where the time of one step goes (ms, median of 20 steps)

| | vanilla AdamW | fused AdamW | FORGE (bf16) | FORGE (int8) |
|---|:--:|:--:|:--:|:--:|
| **Sync after each phase** | | | | |
| Forward | 56.9 | 55.6 | 57.8 | 58.3 |
| Backward : Python returns | 38.0 | 46.5 | 67.4 | 65.8 |
| Backward : TPU finishes | 88.7 | 86.6 | 100.7 | 104.8 |
| Optimizer step | 79.7 | 31.9 | 8.6 | 8.2 |
| Total | 225.3 | 174.2 | 167.1 | 171.3 |
| **As the benchmark (one sync at the end)** | | | | |
| Python done issuing | 98.2 | 83.7 | 93.2 | 100.6 |
| Step time (TPU done) | **231.2** | **168.7** | **162.6** | **165.3** |
| of which FORGE Python | - | - | 16.6 | 17.6 |
| FORGE updates per step | - | - | 112 | 112 |

### Llama-3.2-1B (adamw) : profile in detail (median of 20 steps, after 10 warmup steps)

| Method | Step as benchmark (ms) | Python done issuing (ms) | Forward : Python / TPU done | Backward : Python / TPU done | Step : Python / TPU done | FORGE Python (ms) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla AdamW | 231.2 | 98.2 | 36.1 / 56.9 | 38.0 / 88.7 | 7.6 / 79.7 | 0.0 |
| fused AdamW | 168.7 | 83.7 | 39.2 / 55.6 | 46.5 / 86.6 | 2.3 / 31.9 | 0.0 |
| FORGE (bf16) | 162.6 | 93.2 | 34.4 / 57.8 | 67.4 / 100.7 | 1.4 / 8.6 | 16.6 |
| FORGE (int8) | 165.3 | 100.6 | 34.5 / 58.3 | 65.8 / 104.8 | 1.3 / 8.2 | 17.6 |

Host time per phase (ms, under the profiler), the 3 largest groups :

| Method | Forward | Backward | Step |
|---|---|---|---|
| vanilla AdamW | elementwise 13.4, copies, new tensors 7.0, Python, waiting (no op) 6.8 | elementwise 23.7, copies, new tensors 8.8, autograd engine 7.8 | optimizer 8.5, elementwise 0.3, copies, new tensors 0.2 |
| fused AdamW | elementwise 14.8, copies, new tensors 7.2, Python, waiting (no op) 7.1 | elementwise 20.8, copies, new tensors 8.0, autograd engine 6.8 | optimizer 2.5, copies, new tensors 0.2, Python, waiting (no op) 0.1 |
| FORGE (bf16) | elementwise 15.1, copies, new tensors 7.5, Python, waiting (no op) 7.2 | elementwise 30.4, FORGE update (Python) 21.4, copies, new tensors 13.3 | optimizer 1.2, Python, waiting (no op) 0.1, copies, new tensors 0.1 |
| FORGE (int8) | elementwise 19.8, Python, waiting (no op) 9.6, copies, new tensors 9.6 | elementwise 29.7, FORGE update (Python) 19.4, copies, new tensors 12.9 | optimizer 1.5, copies, new tensors 0.1, Python, waiting (no op) 0.1 |
