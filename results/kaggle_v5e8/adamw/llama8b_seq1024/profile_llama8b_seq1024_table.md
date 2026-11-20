### Llama-3.1-8B (adamw, 8 chips, sequence 1024) : where the time of one step goes (ms, median of 20 steps)

| | vanilla AdamW | fused AdamW | FORGE (bf16) | FORGE (int8) |
|---|:--:|:--:|:--:|:--:|
| **Sync after each phase** | | | | |
| Forward | 111.0 | 103.5 | 109.7 | 106.4 |
| Backward : Python returns | 69.5 | 71.3 | 122.5 | 142.5 |
| Backward : TPU finishes | 157.1 | 146.0 | 173.4 | 172.1 |
| Optimizer step | 66.1 | 30.0 | 6.2 | 6.3 |
| Total | 334.2 | 279.5 | 289.3 | 284.8 |
| **As the benchmark (one sync at the end)** | | | | |
| Python done issuing | 142.2 | 140.8 | 223.4 | 268.2 |
| Step time (TPU done) | **328.4** | **285.0** | **276.8** | **280.6** |
| of which FORGE Python | - | - | 75.3 | 77.1 |
| FORGE updates per step | - | - | 224 | 224 |

### Llama-3.1-8B (adamw, 8 chips, sequence 1024) : profile in detail (median of 20 steps, after 10 warmup steps)

| Method | Step as benchmark (ms) | Python done issuing (ms) | Forward : Python / TPU done | Backward : Python / TPU done | Step : Python / TPU done | FORGE Python (ms) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla AdamW | 328.4 | 142.2 | 56.3 / 111.0 | 69.5 / 157.1 | 20.9 / 66.1 | 0.0 |
| fused AdamW | 285.0 | 140.8 | 66.5 / 103.5 | 71.3 / 146.0 | 5.0 / 30.0 | 0.0 |
| FORGE (bf16) | 276.8 | 223.4 | 65.1 / 109.7 | 122.5 / 173.4 | 1.7 / 6.2 | 75.3 |
| FORGE (int8) | 280.6 | 268.2 | 74.6 / 106.4 | 142.5 / 172.1 | 1.8 / 6.3 | 77.1 |

Host time per phase (ms, under the profiler), the 3 largest groups :

| Method | Forward | Backward | Step |
|---|---|---|---|
| vanilla AdamW | elementwise 26.0, Python, waiting (no op) 13.8, copies, new tensors 13.6 | elementwise 33.2, copies, new tensors 12.7, autograd engine 12.1 | optimizer 19.8, elementwise 0.5, copies, new tensors 0.3 |
| fused AdamW | elementwise 24.5, copies, new tensors 13.3, Python, waiting (no op) 12.6 | elementwise 33.8, copies, new tensors 12.7, autograd engine 11.5 | optimizer 5.1, copies, new tensors 0.3, Python, waiting (no op) 0.1 |
| FORGE (bf16) | elementwise 27.1, Python, waiting (no op) 14.8, copies, new tensors 14.5 | copies, new tensors 47.0, elementwise 33.1, FORGE update (Python) 24.9 | optimizer 1.6, copies, new tensors 0.1, Python, waiting (no op) 0.1 |
| FORGE (int8) | elementwise 31.8, Python, waiting (no op) 17.2, copies, new tensors 16.3 | copies, new tensors 40.4, elementwise 36.2, FORGE update (Python) 29.0 | optimizer 2.0, copies, new tensors 0.1, Python, waiting (no op) 0.1 |
