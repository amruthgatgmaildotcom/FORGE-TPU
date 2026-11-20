import importlib, json, os, statistics, sys, time, torch, warnings
from forge_tpu import _bridge
from config import OPTIM, SEED, LABELS, profile_path

#   Readme.md
#     python benchmarks/profiling.py <model> <method>     (environment : OPTIM, SEQ, see config.py)
#     Where does the time of one training step go ? The same model, data and optimizer as train.py (no lr warmup : time does not depend on the lr).
#     WARMUP steps first (compile), then the same step in two ways, STEPS times each (median), into results/<optim>/<model>/profile_<method>.json :
#        one_sync     forward, backward, step, then ONE sync, as train.py times it. ms from the start of the step : when Python has ISSUED forward, backward, step,
#                     and "done" when the TPU has FINISHED all of it. The TPU works while Python issues the next ops. "forge_python" : Python time inside FORGE's updates
#        per_phase    a sync after each phase : for forward, backward, step, "issued" (ms until Python has issued it) and "done" (ms until the TPU has finished it)
#        host_ms      PROFILED steps under torch.profiler (host side only) : the host time of each phase, in groups (GROUPS). The profiler slows the host down :
#                     the groups show where host time goes, not the real step time
#        forge_updates   FORGE updates per step. replayed_keys / jax_op_keys : launch keys on the fast replay path / on the slow jax_op path (_bridge.py)
WARMUP, STEPS, PROFILED = 10, 20, 3
PHASES = ("forward", "backward", "step")
GROUPS = {"FORGE update (Python)": ("fusedlinearfunction",),                                 # the first group whose pattern is in the op name (lower case) takes the op
          "optimizer":             ("optimizer.step", "fused_adam", "fused_sgd", "_foreach", "addcdiv", "addcmul", "lerp"),
          "matmul":                ("aten::mm", "aten::addmm", "aten::bmm", "aten::matmul", "aten::linear"),
          "attention":             ("scaled_dot_product", "attention"),
          "autograd engine":       ("autograd::", "backward0", "backward1"),
          "copies, new tensors":   ("empty", "zeros", "ones", "full", "fill_", "zero_", "copy_", "_to_copy", "aten::to", "clone", "contiguous", "aten::cat", "embedding", "index", "_backward"),
          "views (no data moved)": ("view", "reshape", "transpose", "permute", "expand", "squeeze", "aten::t", "slice", "select", "as_strided", "chunk", "split", "alias", "detach", "narrow"),
          "elementwise":           ("aten::",)}                                              # every other aten op : mul, add, silu, rsqrt, sum, ...
OTHER = "Python, waiting (no op)"                                                            # phase time inside no op : the model's Python code, autograd waiting for the TPU, ...

def group(name):
    name = name.lower()
    return next((g for g, keys in GROUPS.items() if any(k in name for k in keys)), OTHER)



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def measure(name, method):
    M = importlib.import_module(name)
    sync = torch.accelerator.synchronize
    data = M.make_data(SEED)
    model, opt = M.make_model(method, SEED); model.train()
    forge = [0.0, 0]                                                                         # Python seconds inside FORGE's updates, number of updates
    original = _bridge.fused_update
    def counted(*args, **kwargs):
        t = time.perf_counter(); out = original(*args, **kwargs); forge[0] += time.perf_counter() - t; forge[1] += 1; return out
    _bridge.fused_update = counted                                                           # autograd.py calls _bridge.fused_update at run time, so it calls this one

    def one_sync(x, y, mark = lambda phase: None):
        sync(); forge[0] = 0.0; t0 = time.perf_counter(); t = {}
        mark("forward"); loss = M.loss_fn(model(x), y); t["forward"] = time.perf_counter()
        mark("backward"); loss.backward(); t["backward"] = time.perf_counter()
        mark("step")
        if opt: opt.step(); opt.zero_grad(set_to_none = True)
        t["step"] = time.perf_counter(); mark(None); sync(); t["done"] = time.perf_counter()
        return {k: (v - t0) * 1e3 for k, v in t.items()} | {"forge_python": forge[0] * 1e3}

    def per_phase(x, y):
        r = {}
        sync(); t0 = time.perf_counter(); loss = M.loss_fn(model(x), y); t1 = time.perf_counter(); sync(); t2 = time.perf_counter()
        r["forward"] = {"issued": (t1 - t0) * 1e3, "done": (t2 - t0) * 1e3}
        t0 = time.perf_counter(); loss.backward(); t1 = time.perf_counter(); sync(); t2 = time.perf_counter()
        r["backward"] = {"issued": (t1 - t0) * 1e3, "done": (t2 - t0) * 1e3}
        t0 = time.perf_counter()
        if opt: opt.step(); opt.zero_grad(set_to_none = True)
        t1 = time.perf_counter(); sync(); t2 = time.perf_counter()
        r["step"] = {"issued": (t1 - t0) * 1e3, "done": (t2 - t0) * 1e3}
        return r

    batch = lambda step: data[step % len(data)]
    for step in range(WARMUP): one_sync(*batch(step))
    A = [one_sync(*batch(step)) for step in range(STEPS)]
    B = [per_phase(*batch(step)) for step in range(STEPS)]

    # profiled steps : host time of each phase, by group. A phase = the time between two record_function marks (any thread : the backward can run on another one)
    from torch.profiler import profile, ProfilerActivity, record_function
    host = {p: {} for p in PHASES}
    with profile(activities = [ProfilerActivity.CPU]) as prof:
        for step in range(PROFILED):
            current = []
            def mark(phase):
                if current: current.pop().__exit__(None, None, None)
                if phase: current.append(record_function(f"phase: {phase}")); current[-1].__enter__()
            one_sync(*batch(step), mark)
    events = list(prof.events())
    spans = [(e.name[7:], e.time_range.start, e.time_range.end) for e in events if e.name.startswith("phase: ")]
    for e in events:
        if e.name.startswith("phase: "): continue
        phase = next((p for p, a, b in spans if a <= e.time_range.start < b), None)
        if phase: host[phase][group(e.name)] = host[phase].get(group(e.name), 0) + e.self_cpu_time_total / 1e3 / PROFILED
    for p in PHASES:                                                                         # not inside an op : phase length - time in ops (0 if several threads add up to more)
        length = sum(b - a for q, a, b in spans if q == p) / 1e3 / PROFILED
        host[p][OTHER] = max(length - sum(host[p].values()), 0.0)

    return dict(model = name, optimizer = OPTIM, method = method, seed = SEED, warmup = WARMUP, steps = STEPS, profiled_steps = PROFILED,
                one_sync = {k: statistics.median(a[k] for a in A) for k in A[0]},
                per_phase = {p: {k: statistics.median(b[p][k] for b in B) for k in ("issued", "done")} for p in PHASES},
                host_ms = host, forge_updates = forge[1] // (WARMUP + 2 * STEPS + PROFILED) if forge[1] else 0,
                replayed_keys = len(_bridge._LAUNCHES), jax_op_keys = len(_bridge._NO_REPLAY))

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    name, method = sys.argv[1], sys.argv[2]
    R = measure(name, method)
    path = profile_path(OPTIM, name, method); os.makedirs(os.path.dirname(path), exist_ok = True)
    json.dump(R, open(path, "w"))
    o, s = R["one_sync"], R["per_phase"]
    print(f"{name} {LABELS[method]} : step {o['done']:.1f} ms (Python done issuing at {o['step']:.1f} ms). Sync per phase : "
          + "   ".join(f"{p} {s[p]['issued']:.1f} / {s[p]['done']:.1f}" for p in PHASES) + (f". FORGE : {R['forge_updates']} updates, {o['forge_python']:.1f} ms Python" if R["forge_updates"] else ""), flush = True)
