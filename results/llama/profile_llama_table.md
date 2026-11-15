### Llama-3.2-1B : where the time of one step goes (ms, median of 20 steps)

| | vanilla AdamW | fused AdamW | FORGE (bf16) | FORGE (int8) |
|---|:--:|:--:|:--:|:--:|
| **Sync after each phase** | | | | |
| Forward | 55.8 | 59.3 | 45.9 | 52.9 |
| Backward : Python returns | 50.3 | 46.3 | 87.5 | 70.4 |
| Backward : TPU finishes | 87.1 | 95.0 | 91.5 | 100.2 |
| Optimizer step | 88.3 | 32.7 | 7.7 | 7.9 |
| Total | 231.3 | 186.9 | 145.1 | 161.0 |
| **As the benchmark (one sync at the end)** | | | | |
| Python done issuing | 105.8 | 87.1 | 102.9 | 112.3 |
| Step time (TPU done) | **245.1** | **175.9** | **143.8** | **167.3** |
| of which FORGE Python | - | - | 15.4 | 18.7 |
| FORGE updates per step | - | - | 112 | 112 |

### Llama-3.2-1B : profile in detail (median of 20 steps, after 10 warmup steps)

| Method | Step as benchmark (ms) | Python done issuing (ms) | Forward : Python / TPU done | Backward : Python / TPU done | Step : Python / TPU done | FORGE Python (ms) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla AdamW | 245.1 | 105.8 | 34.8 / 55.8 | 50.3 / 87.1 | 31.0 / 88.3 | 0.0 |
| fused AdamW | 175.9 | 87.1 | 32.3 / 59.3 | 46.3 / 95.0 | 2.3 / 32.7 | 0.0 |
| FORGE (bf16) | 143.8 | 102.9 | 30.0 / 45.9 | 87.5 / 91.5 | 1.2 / 7.7 | 15.4 |
| FORGE (int8) | 167.3 | 112.3 | 32.0 / 52.9 | 70.4 / 100.2 | 1.3 / 7.9 | 18.7 |

Host time per phase (ms, under the profiler), the 3 largest groups :

| Method | Forward | Backward | Step |
|---|---|---|---|
| vanilla AdamW | elementwise 15.9, copies, new tensors 8.6, Python, waiting (no op) 7.1 | elementwise 23.2, copies, new tensors 9.5, autograd engine 7.6 | optimizer 14.5, elementwise 14.3, copies, new tensors 2.7 |
| fused AdamW | elementwise 13.1, copies, new tensors 6.8, Python, waiting (no op) 6.4 | elementwise 16.6, copies, new tensors 6.5, autograd engine 6.0 | optimizer 2.1, copies, new tensors 0.2, Python, waiting (no op) 0.1 |
| FORGE (bf16) | elementwise 14.5, Python, waiting (no op) 7.1, copies, new tensors 6.8 | elementwise 19.6, FORGE update (Python) 13.8, copies, new tensors 8.6 | optimizer 1.1, copies, new tensors 0.1, Python, waiting (no op) 0.1 |
| FORGE (int8) | elementwise 20.8, Python, waiting (no op) 10.0, copies, new tensors 9.8 | elementwise 20.0, FORGE update (Python) 15.0, copies, new tensors 9.0 | optimizer 1.5, copies, new tensors 0.1, Python, waiting (no op) 0.1 |
