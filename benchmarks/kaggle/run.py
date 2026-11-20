import importlib, json, os, sys, time, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); BENCH = os.path.dirname(HERE); REPO = os.path.dirname(BENCH)
sys.path[:0] = [HERE, BENCH, os.path.join(REPO, "src")]
warnings.filterwarnings("ignore")
import torch, torch.distributed as dist
import tp
from config import OPTIM, METHODS, SEED, RESULTS, run_path, profile_path

#   Readme.md
#     One run on the 8 chips, one process per chip (started by kaggle_cell.py with torchrun). Only chip 0 prints and writes files. The environment : see benchmarks/config.py
#     (kaggle_cell.py gives RESULTS = results/kaggle_v5e8, MAX_STEPS = 200, MAX_SECONDS = 1e9 (a FIXED step count : every chip runs the same steps), WARMUP = 10, LR_WARMUP = 0, SEQ = 512)
#        python -m torch.distributed.run --standalone --nproc_per_node 8 benchmarks/kaggle/run.py smoke                  the 8-chip checks         -> <optim>/smoke.md
#        ... benchmarks/kaggle/run.py bench   llama8b <method>       train.py's measure, + the peaks of every chip      -> <optim>/llama8b/<method>_seed<N>.json
#        ... benchmarks/kaggle/run.py profile llama8b <method>       profiling.py's measure                              -> <optim>/llama8b/profile_<method>.json
#     The json of a run is train.py's, with the memory of chip 0 and these additions : params (the FULL model), params_per_chip, chips, peak_per_chip_gib (the timed peak of
#     every chip), memory_gib.peak (the largest of them) and memory_gib.activations from it, peak_synced_gib / peak_synced_per_chip_gib (the peak of the traced step : a sync
#     after every layer, so no memory of ops the host issued ahead, the same kind of number as torch.cuda.max_memory_allocated on a GPU)
if tp.DEVICE == "cpu":
    # only the CPU test of this code (8 processes, gloo) : no accelerator, so its calls are no-ops
    for name in ("synchronize", "reset_peak_memory_stats"): setattr(torch.accelerator, name, lambda *a, **k: None)
    for name in ("memory_allocated", "max_memory_allocated"): setattr(torch.accelerator, name, lambda *a, **k: 0)



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def gather(v): return tp.all_gather(torch.tensor([v], dtype = torch.float32, device = tp.DEVICE)).flatten().tolist()   # one number of every chip

def bench(model, method):
    import train
    M = importlib.import_module(model)
    R = train.measure(model, method)
    peaks, synced = gather(R["memory_gib"]["peak"]), gather(max(p[1] for p in R["trace"]))
    if tp.RANK: return
    g = R["memory_gib"]                                                                      # weights, states, grads : of chip 0 (every chip holds the same sizes)
    g.update(peak = max(peaks), activations = max(peaks) - (g["weights"] + g["states"] + g["grads"]))
    R.update(params_per_chip = R["params"], params = M.PARAMS, chips = tp.WORLD, peak_per_chip_gib = peaks, peak_synced_gib = max(synced), peak_synced_per_chip_gib = synced)
    path = run_path(OPTIM, model, method, SEED); os.makedirs(os.path.dirname(path), exist_ok = True); json.dump(R, open(path, "w"))
    print(f"{model} {method} : step {R['step_ms']['median']:.1f} ± {R['step_ms']['std']:.1f} ms, {R['tflops']['median']:.0f} TF/s (8 chips), peak {g['peak']:.2f} GiB (largest chip; "
          f"synced step {R['peak_synced_gib']:.2f}), {len(R['loss'])} steps -> {path}", flush = True)

def profile(model, method):
    import profiling
    R = profiling.measure(model, method)                                                     # a fixed number of steps : the same on every chip
    if tp.RANK: return
    R["chips"] = tp.WORLD
    path = profile_path(OPTIM, model, method); os.makedirs(os.path.dirname(path), exist_ok = True); json.dump(R, open(path, "w"))

def say(*a):
    if tp.RANK == 0: print(*a, flush = True)



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def smoke():
    # 1. the chips : one device per process, collectives give the right numbers, memory is separate per chip
    lines, ok = [f"# 8-chip checks ({OPTIM})", ""], True
    dev = torch.device(tp.DEVICE)
    s = tp.all_sum(torch.tensor([tp.RANK + 1.0], device = dev)).item()
    g = tp.all_gather(torch.tensor([float(tp.RANK)], device = dev)).flatten().tolist()
    b = tp.all_sum(torch.full((1024,), tp.RANK + 1.0, device = dev, dtype = tp.DTYPE)).float().mean().item()
    want = tp.WORLD * (tp.WORLD + 1) / 2
    collectives = s == want and g == [float(r) for r in range(tp.WORLD)] and b == want
    ok &= collectives
    try: count = torch.accelerator.device_count() if tp.DEVICE != "cpu" else 1
    except Exception as e: count = f"? ({type(e).__name__})"
    before = torch.accelerator.memory_allocated()
    x = torch.empty(256 * 2**20, dtype = torch.uint8, device = dev); torch.accelerator.synchronize() if tp.DEVICE != "cpu" else None
    grew = tp.all_gather(torch.tensor([(torch.accelerator.memory_allocated() - before) / 2**20], device = dev)).flatten().tolist(); del x
    lines += [f"- chips (processes) : {tp.WORLD}, devices seen by each process : {count}",
              f"- all_reduce sum of rank+1 : {s} (want {want}), bf16 : {b}, all_gather of rank : {g} -> {'PASS' if collectives else 'FAIL'}",
              f"- memory of each chip after 256 MiB on each (MiB) : {[round(v) for v in grew]}", ""]
    say("\n".join(lines))

    # 2. TP against one chip : the same small models, weights and data. Every chip also runs the 1-chip model (no collectives). After ONE step, at lr 1e-2 (an update
    #    far above the bf16 rounding of the weights) : (a) every gradient that exists (all of them for the plain torch method, the non-FORGE ones else) against its slice
    #    of the 1-chip gradient, (b) every weight change (FORGE layers too) against its slice of the 1-chip change. Then 3 more steps : the loss of both, for the table
    models = [importlib.import_module("llama8b")]
    rows = ["| model | method | max gradient error | max update error | loss TP (4 steps) | loss 1 chip | replicated weights equal | result |", "|---|---|---|---|---|---|---|---|"]
    cases = [(M, n) for M in models for n in METHODS[OPTIM]]
    for M, name in cases:
        t0 = time.time()
        try:
            gerr, derr, losses, spread = tp_check(M, name)
            good = gerr <= GRAD_TOL and derr <= UPDATE_TOL and spread == 0.0
            ok &= good
            rows.append(f"| {M.TITLE} (small) | {name} | {gerr:.2e} | {derr:.2e} | {' '.join(f'{v:.4f}' for v in losses[True])} | {' '.join(f'{v:.4f}' for v in losses[False])} | "
                        f"{'yes' if spread == 0.0 else f'NO (spread {spread:.2e})'} | {'PASS' if good else 'FAIL'} |")
        except Exception as e:
            ok = False; rows.append(f"| {M.TITLE} (small) | {name} | error : {type(e).__name__} : {str(e)[:300]} | | | | | FAIL |")
            if tp.RANK == 0: import traceback; traceback.print_exc()
        say(rows[-1], f"({time.time() - t0:.0f} s)")
    lines += [f"TP check : each small model in tensor parallel over the 8 chips against the same model on one chip (same weights and data), one step at lr 1e-2. Pass : gradient error "
              f"(max |TP - 1 chip| / max |1 chip|, each tensor) <= {GRAD_TOL}, update error (|change TP - change 1 chip| / |change 1 chip|, each tensor) <= {UPDATE_TOL}, "
              "the replicated (norm) weights equal on all chips.", "", *rows, "", f"**{'ALL PASS' if ok else 'SOME FAIL'}**"]
    if tp.RANK == 0:
        os.makedirs(os.path.join(RESULTS, OPTIM), exist_ok = True); open(os.path.join(RESULTS, OPTIM, "smoke.md"), "w").write("\n".join(lines) + "\n")
        say(f"\n8-chip checks ({OPTIM}) : {'ALL PASS' if ok else 'SOME FAIL'}")

GRAD_TOL, UPDATE_TOL = 5e-2, 0.5                                     # AdamW's first step is lr x sign(g) : bf16 rounding flips the sign of some near-zero g, so the update error of a correct run is ~0.3 (a wrong gradient : > 1)

def tp_check(M, name):
    lr = M.LR; M.LR = 1e-2                                             # build() reads M.LR
    runs = {}
    try:
        for tp_on in (True, False):
            torch.manual_seed(0)
            model, opt = M.build(name, tp_on, M.SMALL); data = small_lm_data(M.SMALL.vocab_size, 128)
            loss = lambda out, y: tp.vocab_cross_entropy(out.logits, y, model.vocab_start, tp_on)
            model.train(); losses = []
            before = {n: p.detach().float().clone() for n, p in model.named_parameters()}
            for step in range(4):
                x, y = data[step % len(data)]
                l = loss(model(x), y); l.backward()
                if step == 0: grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
                if opt: opt.step(); opt.zero_grad(set_to_none = True)
                if step == 0: change = {n: p.detach().float() - before[n] for n, p in model.named_parameters()}
                losses.append(l.item())
            spread = replicated_spread(model) if tp_on else 0.0
            runs[tp_on] = (grads, change, losses, spread)
            del model, opt
    finally:
        M.LR = lr
    (g1, c1, l1, spread), (g8, c8, l8, _) = runs[True], runs[False]
    gerr = max([(g - shard(g8[n], g.shape)).abs().max().item() / max(shard(g8[n], g.shape).abs().max().item(), 1e-30) for n, g in g1.items()] or [0.0])
    moved = any(c.abs().max().item() > 0 for c in c8.values())       # the 1-chip model changed at all (else the update check would compare zeros)
    derr = max([(c - shard(c8[n], c.shape)).norm().item() / max(shard(c8[n], c.shape).norm().item(), 1e-30) for n, c in c1.items()] or [0.0]) if moved else float("inf")
    worst = tp.all_gather(torch.tensor([gerr, derr], device = tp.DEVICE)).max(0).values.tolist()   # the worst chip
    return worst[0], worst[1], {True: l1, False: l8}, spread

def shard(full, shape):
    # this chip's slice of a 1-chip tensor : the same tensor if it is replicated, else the RANK-th part along the dimension that is split
    if tuple(full.shape) == tuple(shape): return full
    d = next(i for i, (a, b) in enumerate(zip(full.shape, shape)) if a != b)
    return full.narrow(d, tp.RANK * shape[d], shape[d])

def small_lm_data(vocab, seq):
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(vocab, (4, 1, seq + 1), generator = g).to(tp.DEVICE)
    return [(b[:, :-1].contiguous(), b[:, 1:].contiguous()) for b in ids]


def replicated_spread(model):
    # the norm weights (the same on every chip) : the largest difference of their sum between the chips, 0.0 if they are equal everywhere
    sums = [p.detach().float().sum() for n, p in model.named_parameters() if "norm" in n]
    if not sums: return 0.0
    g = tp.all_gather(torch.stack(sums))
    return (g.max(0).values - g.min(0).values).abs().max().item()


if __name__ == "__main__":
    tp.init()
    tp.quiet_other_ranks(os.environ.get("RUN_LOGS", os.path.join(RESULTS, "logs")))
    say(f"run.py {' '.join(sys.argv[1:])} : {tp.WORLD} chips, OPTIM={OPTIM}")
    {"smoke": smoke, "bench": bench, "profile": profile}[sys.argv[1]](*sys.argv[2:])
    tp.barrier()
    dist.destroy_process_group()
