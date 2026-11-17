### Llama-3.2-1B (sgd) : where the time of one step goes (ms, median of 20 steps)

| | vanilla SGD | fused SGD | FORGE (SGD) |
|---|:--:|:--:|:--:|
| **Sync after each phase** | | | |
| Forward | 58.2 | 57.4 | 45.7 |
| Backward : Python returns | 37.7 | 49.6 | 64.4 |
| Backward : TPU finishes | 92.5 | 90.1 | 72.0 |
| Optimizer step | 26.3 | 14.5 | 4.1 |
| Total | 177.0 | 162.0 | 121.8 |
| **As the benchmark (one sync at the end)** | | | |
| Python done issuing | 81.5 | 89.4 | 105.3 |
| Step time (TPU done) | **172.4** | **139.2** | **126.3** |
| of which FORGE Python | - | - | 14.6 |
| FORGE updates per step | - | - | 112 |

### Llama-3.2-1B (sgd) : profile in detail (median of 20 steps, after 10 warmup steps)

| Method | Step as benchmark (ms) | Python done issuing (ms) | Forward : Python / TPU done | Backward : Python / TPU done | Step : Python / TPU done | FORGE Python (ms) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla SGD | 172.4 | 81.5 | 42.6 / 58.2 | 37.7 / 92.5 | 2.5 / 26.3 | 0.0 |
| fused SGD | 139.2 | 89.4 | 45.8 / 57.4 | 49.6 / 90.1 | 1.1 / 14.5 | 0.0 |
| FORGE (SGD) | 126.3 | 105.3 | 35.2 / 45.7 | 64.4 / 72.0 | 0.9 / 4.1 | 14.6 |

Host time per phase (ms, under the profiler), the 3 largest groups :

| Method | Forward | Backward | Step |
|---|---|---|---|
| vanilla SGD | elementwise 15.8, copies, new tensors 7.9, Python, waiting (no op) 6.8 | elementwise 18.6, copies, new tensors 7.2, autograd engine 6.9 | optimizer 2.2, copies, new tensors 0.2, Python, waiting (no op) 0.0 |
| fused SGD | elementwise 18.0, copies, new tensors 8.9, Python, waiting (no op) 7.8 | elementwise 26.3, copies, new tensors 9.9, autograd engine 8.2 | optimizer 1.1, copies, new tensors 0.2, Python, waiting (no op) 0.1 |
| FORGE (SGD) | elementwise 16.0, Python, waiting (no op) 7.8, copies, new tensors 7.7 | elementwise 19.9, FORGE update (Python) 12.0, copies, new tensors 8.3 | optimizer 0.8, Python, waiting (no op) 0.1, copies, new tensors 0.1 |
