# 8-chip checks (muon)

- chips (processes) : 8, devices seen by each process : ? (RuntimeError)
- all_reduce sum of rank+1 : 36.0 (want 36.0), bf16 : 36.0, all_gather of rank : [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] -> PASS
- memory of each chip after 256 MiB on each (MiB) : [0, 0, 0, 0, 0, 0, 0, 0]

TP check : each small model in tensor parallel over the 8 chips against the same model on one chip (same weights and data), one step at lr 1e-2. Pass : gradient error (max |TP - 1 chip| / max |1 chip|, each tensor) <= 0.05, update error (|change TP - change 1 chip| / |change 1 chip|, each tensor) <= 0.5, the replicated (norm) weights equal on all chips.

| model | method | max gradient error | max update error | loss TP (4 steps) | loss 1 chip | replicated weights equal | result |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B (small) | muon | 1.73e-02 | 2.72e-01 | 7.6206 7.6911 7.7116 7.7372 | 7.6203 7.6911 7.7139 7.7414 | yes | PASS |
| Llama-3.1-8B (small) | forge_muon | 1.69e-02 | 2.72e-01 | 7.6206 7.6911 7.7123 7.7373 | 7.6203 7.6911 7.7141 7.7399 | yes | PASS |

**ALL PASS**
