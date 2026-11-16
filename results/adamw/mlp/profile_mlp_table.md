### MLP (adamw) : where the time of one step goes (ms, median of 20 steps)

| | vanilla AdamW | fused AdamW | FORGE (bf16) | FORGE (int8) |
|---|:--:|:--:|:--:|:--:|
| **Sync after each phase** | | | | |
| Forward | 68.7 | 68.7 | 68.7 | 68.6 |
| Backward : Python returns | 1.4 | 1.3 | 3.5 | 3.7 |
| Backward : TPU finishes | 123.6 | 123.6 | 118.1 | 119.1 |
| Optimizer step | 67.3 | 25.1 | 0.1 | 0.1 |
| Total | 259.5 | 217.4 | 186.8 | 187.9 |
| **As the benchmark (one sync at the end)** | | | | |
| Python done issuing | 3.4 | 3.0 | 5.0 | 4.8 |
| Step time (TPU done) | **257.8** | **215.4** | **186.3** | **187.4** |
| of which FORGE Python | - | - | 1.7 | 1.7 |
| FORGE updates per step | - | - | 17 | 17 |

### MLP (adamw) : profile in detail (median of 20 steps, after 10 warmup steps)

| Method | Step as benchmark (ms) | Python done issuing (ms) | Forward : Python / TPU done | Backward : Python / TPU done | Step : Python / TPU done | FORGE Python (ms) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla AdamW | 257.8 | 3.4 | 1.0 / 68.7 | 1.4 / 123.6 | 1.1 / 67.3 | 0.0 |
| fused AdamW | 215.4 | 3.0 | 1.1 / 68.7 | 1.3 / 123.6 | 0.4 / 25.1 | 0.0 |
| FORGE (bf16) | 186.3 | 5.0 | 1.2 / 68.7 | 3.5 / 118.1 | 0.0 / 0.1 | 1.7 |
| FORGE (int8) | 187.4 | 4.8 | 1.5 / 68.6 | 3.7 / 119.1 | 0.0 / 0.1 | 1.7 |

Host time per phase (ms, under the profiler), the 3 largest groups :

| Method | Forward | Backward | Step |
|---|---|---|---|
| vanilla AdamW | elementwise 0.6, Python, waiting (no op) 0.4, matmul 0.3 | Python, waiting (no op) 0.6, copies, new tensors 0.6, autograd engine 0.5 | optimizer 1.0, copies, new tensors 0.1, Python, waiting (no op) 0.0 |
| fused AdamW | elementwise 0.6, Python, waiting (no op) 0.4, matmul 0.3 | Python, waiting (no op) 0.5, copies, new tensors 0.4, autograd engine 0.3 | optimizer 0.3, Python, waiting (no op) 0.0, copies, new tensors 0.0 |
| FORGE (bf16) | elementwise 0.6, Python, waiting (no op) 0.5, matmul 0.2 | FORGE update (Python) 1.6, copies, new tensors 0.9, Python, waiting (no op) 0.6 | Python, waiting (no op) 0.0 |
| FORGE (int8) | elementwise 0.6, Python, waiting (no op) 0.5, matmul 0.3 | FORGE update (Python) 1.6, copies, new tensors 0.9, Python, waiting (no op) 0.5 | Python, waiting (no op) 0.0 |
