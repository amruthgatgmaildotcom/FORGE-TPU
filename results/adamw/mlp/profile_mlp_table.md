### MLP (adamw) : where the time of one step goes (ms, median of 20 steps)

| | vanilla AdamW | fused AdamW | FORGE (bf16) | FORGE (int8) |
|---|:--:|:--:|:--:|:--:|
| **Sync after each phase** | | | | |
| Forward | 68.8 | 68.7 | 68.7 | 68.7 |
| Backward : Python returns | 1.5 | 1.5 | 3.3 | 3.7 |
| Backward : TPU finishes | 123.6 | 123.5 | 118.1 | 119.2 |
| Optimizer step | 67.4 | 25.2 | 0.1 | 0.1 |
| Total | 259.8 | 217.5 | 186.8 | 188.0 |
| **As the benchmark (one sync at the end)** | | | | |
| Python done issuing | 3.5 | 2.7 | 4.8 | 5.1 |
| Step time (TPU done) | **257.8** | **215.5** | **186.2** | **187.3** |
| of which FORGE Python | - | - | 1.6 | 1.7 |
| FORGE updates per step | - | - | 17 | 17 |

### MLP (adamw) : profile in detail (median of 20 steps, after 10 warmup steps)

| Method | Step as benchmark (ms) | Python done issuing (ms) | Forward : Python / TPU done | Backward : Python / TPU done | Step : Python / TPU done | FORGE Python (ms) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla AdamW | 257.8 | 3.5 | 1.2 / 68.8 | 1.5 / 123.6 | 1.2 / 67.4 | 0.0 |
| fused AdamW | 215.5 | 2.7 | 1.0 / 68.7 | 1.5 / 123.5 | 0.4 / 25.2 | 0.0 |
| FORGE (bf16) | 186.2 | 4.8 | 1.7 / 68.7 | 3.3 / 118.1 | 0.0 / 0.1 | 1.6 |
| FORGE (int8) | 187.3 | 5.1 | 1.6 / 68.7 | 3.7 / 119.2 | 0.0 / 0.1 | 1.7 |

Host time per phase (ms, under the profiler), the 3 largest groups :

| Method | Forward | Backward | Step |
|---|---|---|---|
| vanilla AdamW | elementwise 0.5, Python, waiting (no op) 0.4, matmul 0.2 | Python, waiting (no op) 0.6, copies, new tensors 0.5, autograd engine 0.4 | optimizer 0.8, copies, new tensors 0.1, Python, waiting (no op) 0.0 |
| fused AdamW | elementwise 0.6, Python, waiting (no op) 0.4, matmul 0.2 | Python, waiting (no op) 0.5, copies, new tensors 0.5, autograd engine 0.4 | optimizer 0.3, Python, waiting (no op) 0.0, copies, new tensors 0.0 |
| FORGE (bf16) | elementwise 0.7, Python, waiting (no op) 0.6, matmul 0.3 | FORGE update (Python) 1.4, copies, new tensors 0.9, Python, waiting (no op) 0.6 | Python, waiting (no op) 0.0 |
| FORGE (int8) | elementwise 0.6, Python, waiting (no op) 0.4, matmul 0.3 | FORGE update (Python) 1.6, copies, new tensors 0.9, Python, waiting (no op) 0.6 | Python, waiting (no op) 0.0 |
