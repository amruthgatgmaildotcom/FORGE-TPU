### MLP : where the time of one step goes (ms, median of 20 steps)

| | vanilla AdamW | fused AdamW | FORGE (bf16) | FORGE (int8) |
|---|:--:|:--:|:--:|:--:|
| **Sync after each phase** | | | | |
| Forward | 68.6 | 68.7 | 68.8 | 68.7 |
| Backward : Python returns | 1.6 | 1.5 | 4.1 | 4.5 |
| Backward : TPU finishes | 123.6 | 123.6 | 125.8 | 125.9 |
| Optimizer step | 66.9 | 25.2 | 0.1 | 0.1 |
| Total | 259.1 | 217.5 | 194.7 | 194.6 |
| **As the benchmark (one sync at the end)** | | | | |
| Python done issuing | 5.5 | 3.2 | 5.1 | 4.6 |
| Step time (TPU done) | **257.8** | **215.5** | **193.9** | **194.1** |
| of which FORGE Python | - | - | 1.8 | 1.8 |
| FORGE updates per step | - | - | 17 | 17 |

### MLP : profile in detail (median of 20 steps, after 10 warmup steps)

| Method | Step as benchmark (ms) | Python done issuing (ms) | Forward : Python / TPU done | Backward : Python / TPU done | Step : Python / TPU done | FORGE Python (ms) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla AdamW | 257.8 | 5.5 | 1.1 / 68.6 | 1.6 / 123.6 | 3.0 / 66.9 | 0.0 |
| fused AdamW | 215.5 | 3.2 | 1.5 / 68.7 | 1.5 / 123.6 | 0.5 / 25.2 | 0.0 |
| FORGE (bf16) | 193.9 | 5.1 | 1.3 / 68.8 | 4.1 / 125.8 | 0.0 / 0.1 | 1.8 |
| FORGE (int8) | 194.1 | 4.6 | 1.4 / 68.7 | 4.5 / 125.9 | 0.0 / 0.1 | 1.8 |

Host time per phase (ms, under the profiler), the 3 largest groups :

| Method | Forward | Backward | Step |
|---|---|---|---|
| vanilla AdamW | elementwise 0.5, Python, waiting (no op) 0.4, matmul 0.3 | Python, waiting (no op) 0.6, copies, new tensors 0.4, autograd engine 0.3 | elementwise 1.5, optimizer 1.5, copies, new tensors 0.3 |
| fused AdamW | elementwise 0.7, Python, waiting (no op) 0.4, matmul 0.3 | Python, waiting (no op) 0.6, copies, new tensors 0.5, autograd engine 0.4 | optimizer 0.3, Python, waiting (no op) 0.0, copies, new tensors 0.0 |
| FORGE (bf16) | elementwise 0.6, Python, waiting (no op) 0.5, matmul 0.2 | FORGE update (Python) 1.6, copies, new tensors 0.8, Python, waiting (no op) 0.6 | Python, waiting (no op) 0.0 |
| FORGE (int8) | elementwise 0.5, Python, waiting (no op) 0.5, matmul 0.2 | FORGE update (Python) 1.7, copies, new tensors 0.9, Python, waiting (no op) 0.6 | Python, waiting (no op) 0.0 |
