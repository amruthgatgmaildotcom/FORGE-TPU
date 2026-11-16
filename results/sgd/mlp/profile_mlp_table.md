### MLP (sgd) : where the time of one step goes (ms, median of 20 steps)

| | vanilla SGD | fused SGD | FORGE (SGD) |
|---|:--:|:--:|:--:|
| **Sync after each phase** | | | |
| Forward | 68.7 | 68.7 | 68.7 |
| Backward : Python returns | 1.3 | 1.3 | 3.2 |
| Backward : TPU finishes | 123.6 | 123.6 | 108.0 |
| Optimizer step | 21.0 | 11.0 | 0.1 |
| Total | 213.2 | 203.3 | 176.8 |
| **As the benchmark (one sync at the end)** | | | |
| Python done issuing | 3.1 | 2.7 | 4.4 |
| Step time (TPU done) | **211.6** | **201.7** | **176.3** |
| of which FORGE Python | - | - | 1.3 |
| FORGE updates per step | - | - | 17 |

### MLP (sgd) : profile in detail (median of 20 steps, after 10 warmup steps)

| Method | Step as benchmark (ms) | Python done issuing (ms) | Forward : Python / TPU done | Backward : Python / TPU done | Step : Python / TPU done | FORGE Python (ms) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla SGD | 211.6 | 3.1 | 1.1 / 68.7 | 1.3 / 123.6 | 0.4 / 21.0 | 0.0 |
| fused SGD | 201.7 | 2.7 | 1.3 / 68.7 | 1.3 / 123.6 | 0.3 / 11.0 | 0.0 |
| FORGE (SGD) | 176.3 | 4.4 | 1.4 / 68.7 | 3.2 / 108.0 | 0.0 / 0.1 | 1.3 |

Host time per phase (ms, under the profiler), the 3 largest groups :

| Method | Forward | Backward | Step |
|---|---|---|---|
| vanilla SGD | elementwise 0.5, Python, waiting (no op) 0.4, matmul 0.2 | Python, waiting (no op) 0.5, copies, new tensors 0.5, autograd engine 0.4 | optimizer 0.3, Python, waiting (no op) 0.0, copies, new tensors 0.0 |
| fused SGD | elementwise 0.6, Python, waiting (no op) 0.4, matmul 0.3 | Python, waiting (no op) 0.7, autograd engine 0.5, copies, new tensors 0.5 | optimizer 0.1, Python, waiting (no op) 0.0, copies, new tensors 0.0 |
| FORGE (SGD) | elementwise 0.8, Python, waiting (no op) 0.6, matmul 0.3 | FORGE update (Python) 1.5, Python, waiting (no op) 0.7, copies, new tensors 0.7 | Python, waiting (no op) 0.0 |
