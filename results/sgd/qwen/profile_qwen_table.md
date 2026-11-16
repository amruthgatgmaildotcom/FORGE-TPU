### Qwen3-0.6B (sgd) : where the time of one step goes (ms, median of 20 steps)

| | vanilla SGD | fused SGD | FORGE (SGD) |
|---|:--:|:--:|:--:|
| **Sync after each phase** | | | |
| Forward | 118.0 | 119.8 | 119.6 |
| Backward : Python returns | 126.3 | 149.2 | 136.8 |
| Backward : TPU finishes | 213.2 | 224.0 | 221.9 |
| Optimizer step | 17.3 | 12.4 | 6.2 |
| Total | 348.5 | 356.2 | 347.6 |
| **As the benchmark (one sync at the end)** | | | |
| Python done issuing | 201.7 | 199.2 | 229.5 |
| Step time (TPU done) | **352.1** | **337.1** | **341.5** |
| of which FORGE Python | - | - | 26.8 |
| FORGE updates per step | - | - | 196 |

### Qwen3-0.6B (sgd) : profile in detail (median of 20 steps, after 10 warmup steps)

| Method | Step as benchmark (ms) | Python done issuing (ms) | Forward : Python / TPU done | Backward : Python / TPU done | Step : Python / TPU done | FORGE Python (ms) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla SGD | 352.1 | 201.7 | 62.8 / 118.0 | 126.3 / 213.2 | 5.8 / 17.3 | 0.0 |
| fused SGD | 337.1 | 199.2 | 91.9 / 119.8 | 149.2 / 224.0 | 2.5 / 12.4 | 0.0 |
| FORGE (SGD) | 341.5 | 229.5 | 66.2 / 119.6 | 136.8 / 221.9 | 2.3 / 6.2 | 26.8 |

Host time per phase (ms, under the profiler), the 3 largest groups :

| Method | Forward | Backward | Step |
|---|---|---|---|
| vanilla SGD | elementwise 30.2, copies, new tensors 16.2, Python, waiting (no op) 14.3 | elementwise 59.1, copies, new tensors 21.2, autograd engine 15.7 | optimizer 5.5, copies, new tensors 0.3, Python, waiting (no op) 0.1 |
| fused SGD | elementwise 40.9, copies, new tensors 21.9, Python, waiting (no op) 19.1 | elementwise 68.8, copies, new tensors 25.9, autograd engine 19.1 | optimizer 2.8, copies, new tensors 0.4, Python, waiting (no op) 0.1 |
| FORGE (SGD) | elementwise 30.0, copies, new tensors 16.3, Python, waiting (no op) 15.5 | elementwise 63.0, copies, new tensors 23.9, FORGE update (Python) 23.2 | optimizer 2.2, copies, new tensors 0.1, Python, waiting (no op) 0.1 |
