import gc, importlib, json, os, statistics, sys, time, torch
from forge_tpu import FusedLinear
from config import OPTIM, SEED, MAX_STEPS, MAX_SECONDS, WARMUP, LR_WARMUP, LABELS, run_path
from methods import lr_warmup

#   Readme.md
#     python benchmarks/train.py <model> <method>     (environment : OPTIM, SEED, SEQ, MAX_SECONDS, see config.py)
#     Trains ONE method on ONE model (benchmarks/<model>.py) and measures every step, into results/<optim>/<model>/<method>_seed<N>.json :
#        model, optimizer, method, seed, params, settings      what ran (settings : the model's SETTINGS + warmup, lr_warmup, max_steps, max_seconds)
#        loss, time_ms, peak_gib, held_gib                     one value per step. time : forward + backward + update, between two device syncs.
#                                                              peak : the largest memory inside the step, held : what is still allocated after it (both above the data)
#        trace                                                 one more step (not timed), [held, peak since the point before, label] after every leaf module in the forward,
#                                                              before each of their backwards, and at "start", "loss", "backward", "step"
#        memory_gib   peak, weights, states, grads, activations      peak : the largest step peak after the warmup. grads : the .grad of the torch optimizer's params
#                                                                     (FORGE layers never have one). activations = peak - (weights + states + grads)
#        step_ms, tflops   median, std                         over the steps after the WARMUP first ones. TF/s = the model's flops of one step / step time
GiB = 2**30



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def nbytes(tensors): return sum(t.numel() * t.element_size() for t in tensors)

def state_bytes(model, opt):
    # torch optimizer states + FORGE states
    torch_states = nbytes(t for s in opt.state.values() for t in s.values() if torch.is_tensor(t)) if opt else 0
    return torch_states + nbytes(t for m in model.modules() if isinstance(m, FusedLinear) for arrays in m.fusedStateObject.buffers.values() for t in arrays.values())

def layer_trace(model, opt, batch, loss_fn, base):
    # one more step : memory after every leaf module in the forward, before each of their backwards, and at loss, end of backward, step.
    # Each point is (memory held at the point, peak since the point before) : the peak also shows short-lived tensors (inside an op, the loss, the step)
    acc, points, hooks = torch.accelerator, [], []
    def mark(label = ""):
        acc.synchronize(); points.append(((acc.memory_allocated() - base) / GiB, (acc.max_memory_allocated() - base) / GiB, label)); acc.reset_peak_memory_stats()
    def after_forward(module, inputs, out):
        mark()
        if torch.is_tensor(out) and out.requires_grad: hooks.append(out.register_hook(lambda grad: mark()))   # a tuple (RoPE's cos, sin) or a tensor with no grad : no backward point
    hooks += [m.register_forward_hook(after_forward) for m in model.modules() if not any(m.children())]
    x, y = batch
    mark(); points.clear()                                                                   # starts the peak count
    mark("start"); loss = loss_fn(model(x), y); mark("loss"); loss.backward(); mark("backward")
    if opt: opt.step(); opt.zero_grad(set_to_none = True)
    mark("step")
    for h in hooks: h.remove()
    return points



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def measure(name, method):
    M = importlib.import_module(name)                                                        # benchmarks/<name>.py
    acc = torch.accelerator
    data = M.make_data(SEED)
    acc.synchronize(); base = acc.memory_allocated()                                         # memory is counted above the data
    model, opt = M.make_model(method, SEED)
    model.train()
    set_lr = lr_warmup(model, opt)
    loss, time_ms, peak_gib, held_gib, start = [], [], [], [], time.perf_counter()
    for step in range(MAX_STEPS):
        x, y = data[step % len(data)]
        set_lr(step)
        acc.reset_peak_memory_stats(); t0 = time.perf_counter()
        l = M.loss_fn(model(x), y)
        l.backward()                                                                         # FORGE : the fused weights are updated here, inside backward
        if opt: opt.step(); opt.zero_grad(set_to_none = True)
        acc.synchronize(); t1 = time.perf_counter()
        loss.append(l.item()); time_ms.append((t1 - t0) * 1e3)
        peak_gib.append((acc.max_memory_allocated() - base) / GiB); held_gib.append((acc.memory_allocated() - base) / GiB)
        if step % 20 == 0: print(f"{name} {method:12s} step {step:4d}  loss {loss[-1]:.4f}  {time_ms[-1]:8.1f} ms  {peak_gib[-1]:6.2f} GiB", flush = True)
        if t1 - start > MAX_SECONDS: break
    trace = layer_trace(model, opt, data[0], M.loss_fn, base)
    params, weights, states = sum(p.numel() for p in model.parameters()), nbytes(model.parameters()), state_bytes(model, opt)          # bytes
    grads = nbytes(p for group in opt.param_groups for p in group["params"]) if opt else 0                                          # only the params of a torch optimizer get a .grad
    w = WARMUP if len(time_ms) > WARMUP + 1 else 0                                           # a very short run : every step is timed
    peak, tflops = max(peak_gib[w:]), [M.flops(params) / t / 1e9 for t in time_ms[w:]]      # t in ms
    del model, opt; gc.collect()
    return dict(model = name, optimizer = OPTIM, method = method, seed = SEED, params = params,
                settings = {**M.SETTINGS, "warmup": w, "lr_warmup": LR_WARMUP, "max_steps": MAX_STEPS, "max_seconds": MAX_SECONDS},
                loss = loss, time_ms = time_ms, peak_gib = peak_gib, held_gib = held_gib, trace = trace,
                memory_gib = dict(peak = peak, weights = weights / GiB, states = states / GiB, grads = grads / GiB, activations = peak - (weights + states + grads) / GiB),
                step_ms = dict(median = statistics.median(time_ms[w:]), std = statistics.pstdev(time_ms[w:])),
                tflops = dict(median = statistics.median(tflops), std = statistics.pstdev(tflops)))

if __name__ == "__main__":
    name, method = sys.argv[1], sys.argv[2]
    R = measure(name, method)
    path = run_path(OPTIM, name, method, SEED); os.makedirs(os.path.dirname(path), exist_ok = True)
    json.dump(R, open(path, "w"))
    print(f"{name} {LABELS[method]} (seed {SEED}) : peak {R['memory_gib']['peak']:.2f} GiB, step {R['step_ms']['median']:.1f} ± {R['step_ms']['std']:.1f} ms, "
          f"{R['tflops']['median']:.0f} TF/s, {len(R['loss'])} steps -> {path}", flush = True)
