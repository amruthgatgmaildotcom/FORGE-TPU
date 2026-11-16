### Llama-3.2-1B (sgd) : where the time of one step goes (ms, median of 20 steps)

| | vanilla SGD | fused SGD | FORGE (SGD) |
|---|:--:|:--:|:--:|
| **Sync after each phase** | | | |
| Forward | 57.4 | 57.3 | 58.8 |
| Backward : Python returns | 46.5 | 34.8 | 76.5 |
| Backward : TPU finishes | 90.9 | 91.2 | 98.2 |
| Optimizer step | 25.8 | 14.5 | 4.6 |
| Total | 174.1 | 162.9 | 161.6 |
| **As the benchmark (one sync at the end)** | | | |
| Python done issuing | 80.0 | 78.3 | 111.9 |
| Step time (TPU done) | **169.8** | **159.1** | **155.3** |
| of which FORGE Python | - | - | 17.7 |
| FORGE updates per step | - | - | 112 |

### Llama-3.2-1B (sgd) : profile in detail (median of 20 steps, after 10 warmup steps)

| Method | Step as benchmark (ms) | Python done issuing (ms) | Forward : Python / TPU done | Backward : Python / TPU done | Step : Python / TPU done | FORGE Python (ms) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla SGD | 169.8 | 80.0 | 40.7 / 57.4 | 46.5 / 90.9 | 2.1 / 25.8 | 0.0 |
| fused SGD | 159.1 | 78.3 | 26.6 / 57.3 | 34.8 / 91.2 | 1.1 / 14.5 | 0.0 |
| FORGE (SGD) | 155.3 | 111.9 | 43.7 / 58.8 | 76.5 / 98.2 | 0.9 / 4.6 | 17.7 |

Host time per phase (ms, under the profiler), the 3 largest groups :

| Method | Forward | Backward | Step |
|---|---|---|---|
| vanilla SGD | elementwise 15.3, copies, new tensors 7.6, Python, waiting (no op) 7.3 | elementwise 20.3, copies, new tensors 7.8, autograd engine 7.1 | optimizer 2.7, copies, new tensors 0.2, Python, waiting (no op) 0.1 |
| fused SGD | elementwise 13.7, copies, new tensors 6.8, Python, waiting (no op) 6.7 | elementwise 18.1, copies, new tensors 7.0, autograd engine 6.4 | optimizer 1.0, copies, new tensors 0.2, Python, waiting (no op) 0.1 |
| FORGE (SGD) | elementwise 20.0, copies, new tensors 9.4, Python, waiting (no op) 8.9 | elementwise 28.6, FORGE update (Python) 16.5, copies, new tensors 12.0 | optimizer 0.8, copies, new tensors 0.1, Python, waiting (no op) 0.1 |
