### Qwen3-0.6B (sgd) : where the time of one step goes (ms, median of 20 steps)

| | vanilla SGD | fused SGD | FORGE (SGD) |
|---|:--:|:--:|:--:|
| **Sync after each phase** | | | |
| Forward | 112.0 | 123.2 | 123.0 |
| Backward : Python returns | 112.1 | 111.5 | 157.5 |
| Backward : TPU finishes | 207.7 | 215.6 | 227.2 |
| Optimizer step | 18.4 | 11.1 | 6.1 |
| Total | 338.1 | 349.9 | 356.4 |
| **As the benchmark (one sync at the end)** | | | |
| Python done issuing | 175.7 | 192.0 | 243.9 |
| Step time (TPU done) | **359.6** | **399.0** | **354.5** |
| of which FORGE Python | - | - | 26.4 |
| FORGE updates per step | - | - | 196 |

### Qwen3-0.6B (sgd) : profile in detail (median of 20 steps, after 10 warmup steps)

| Method | Step as benchmark (ms) | Python done issuing (ms) | Forward : Python / TPU done | Backward : Python / TPU done | Step : Python / TPU done | FORGE Python (ms) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla SGD | 359.6 | 175.7 | 72.7 / 112.0 | 112.1 / 207.7 | 6.3 / 18.4 | 0.0 |
| fused SGD | 399.0 | 192.0 | 64.8 / 123.2 | 111.5 / 215.6 | 2.7 / 11.1 | 0.0 |
| FORGE (SGD) | 354.5 | 243.9 | 75.1 / 123.0 | 157.5 / 227.2 | 2.1 / 6.1 | 26.4 |

Host time per phase (ms, under the profiler), the 3 largest groups :

| Method | Forward | Backward | Step |
|---|---|---|---|
| vanilla SGD | elementwise 42.6, copies, new tensors 22.6, Python, waiting (no op) 21.5 | elementwise 67.7, copies, new tensors 25.3, autograd engine 18.0 | optimizer 8.4, copies, new tensors 0.4, Python, waiting (no op) 0.1 |
| fused SGD | elementwise 36.8, copies, new tensors 19.4, Python, waiting (no op) 17.4 | elementwise 84.5, copies, new tensors 30.3, autograd engine 20.3 | optimizer 3.3, copies, new tensors 0.5, Python, waiting (no op) 0.1 |
| FORGE (SGD) | elementwise 28.8, Python, waiting (no op) 15.8, copies, new tensors 15.7 | elementwise 78.6, copies, new tensors 30.6, FORGE update (Python) 27.1 | optimizer 2.6, copies, new tensors 0.2, Python, waiting (no op) 0.1 |
