"""dk4248's metrics (peak memory, step time, TF/s, his two figures) for an MLP on one TPU v5e, like his
examples/quickstart.py (nn.Linear layers with GELU between them, every Linear swapped for a FusedLinear), but sized so
that the weight gradient matters: by default 12 layers of 8192 x 8192 (+ bias) = 805,404,672 parameters.

Why an MLP: every number can be predicted by hand, and the table prints the prediction next to the measurement.
With L layers of width d, T rows per batch (one row = one "token"), bf16 (2 bytes):
    weights           L (d² + d) · 2 B                                1.50 GiB by default
    m + v (bf16)      2 × weights                                     3.00 GiB
    m + v (int8)      L d² · 2 · (1 + 4/64) B  + biases in bf16        1.59 GiB
    dW pool           = weights for vanilla / fused AdamW;  FORGE: only the biases (L d · 2 B)
    activations       (2L - 1) T d · 2 B  (each Linear saves its input, each GELU its input)  +  T d · 4 B (the loss, fp32)
So FORGE should save about the whole dW pool while it is larger than the activations (small T), and nothing once the
activations are larger (T = 8192 by default): dk4248's "operating regime", measured.

The data: a fresh batch every step (seeded by the step number, the same for every method): X ~ N(0, 1) and the target
X @ U @ V, a fixed random rank-64 linear "teacher". The loss is the mean squared error. The loss can only fall by
learning the teacher's function: no batch is ever seen twice, so nothing can be memorized.

The methods, timing, peak and the layer-by-layer step are exactly benchmarks/llama.py's:
    vanilla AdamW   torch.optim.AdamW, default implementation, bf16 weights and states
    fused AdamW     torch.optim.AdamW(fused=True), bf16 weights and states
    FORGE (bf16)    every nn.Linear a FusedLinear (bf16 states); the biases on fused AdamW
    FORGE (int8)    the same with int8 states
The clock starts before the forward and stops after torch.accelerator.synchronize() following forward + loss,
backward, optimizer.step(), optimizer.zero_grad(). The batch is made on the TPU before the clock starts.

COMPILE_TORCH=1: the model and optimizer.step() are compiled with torch.compile(backend="tpu"). TorchTPU 0.2.0 drops
writes into a graph's inputs that are not graph outputs (torch_tpu/_internal/export/export.py, fx_to_mlir: the graph is
run on clones of its inputs and only its outputs are compiled), and FORGE writes its weights and states inside
backward. So FusedLinear.forward is NOT traced (torch.compiler.disable): each FusedLinear runs FORGE's own eager code
(donate + set_), everything between them is compiled. The diagnostics check this: graph breaks = FusedLinear calls,
and FORGE's step counter = the steps run. COMPILE_TORCH=0: eager, optimized (TORCH_TPU_DEFER_AND_FUSE=1).

Correctness: before the benchmark, a small MLP (4 layers of 1024, 256 rows, 10 steps) is trained by each method in
bf16 and by torch AdamW in fp32 (the reference), from the same weights on the same batches; the table gives how far
each method's weights end from the reference, relative to how far the reference moved.

    PYTHONPATH=src python benchmarks/mlp.py          the check, then every method x every batch size
    PYTHONPATH=src python benchmarks/mlp.py plot     only redraws the tables and figures
Knobs (environment variables): COMPILE_TORCH (1/0), BATCH_SIZES (512,2048,8192), MLP_WIDTH (8192), MLP_LAYERS (12),
WARMUP_STEPS (10), TIMED_STEPS (20), METHODS (";"-separated), OUT_DIR (benchmarks/mlp_results), CHECK (1/0).
"""
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ALL_METHODS = ["vanilla AdamW", "fused AdamW", "FORGE (bf16)", "FORGE (int8)"]
ARROWS = [("vanilla AdamW", "FORGE (bf16)"), ("fused AdamW", "FORGE (int8)")]  # the red arrows of figure (b)
LR = 1e-3  # as in dk4248's examples/quickstart.py; betas 0.9 / 0.999, eps 1e-8, weight decay 0.01 on both sides
RANK = 64  # the teacher's rank
GIB = 2**30
CHIP_LIMIT_GIB = 15.5  # a peak above this is the v5e's 16 GiB HBM being full, not a measured need


def knobs():
    env = os.environ.get
    return {"compile": env("COMPILE_TORCH", "1") == "1",
            "batch_sizes": [int(s) for s in env("BATCH_SIZES", "512,2048,8192").split(",") if s.strip()],
            "width": int(env("MLP_WIDTH", "8192")), "layers": int(env("MLP_LAYERS", "12")),
            "warmup": int(env("WARMUP_STEPS", "10")), "timed": int(env("TIMED_STEPS", "20")),
            "methods": [m.strip() for m in env("METHODS", ";".join(ALL_METHODS)).split(";") if m.strip()],
            "out": Path(env("OUT_DIR", str(Path(__file__).resolve().parent / "mlp_results"))),
            "check": env("CHECK", "1") == "1"}


def mode_dir(k):
    return k["out"] / ("compiled" if k["compile"] else "eager")


def out_dir(k, rows):
    return mode_dir(k) / f"batch_{rows}"


def say(*parts):
    print(*parts, flush=True)


# ---- 1. The model, the data, the optimizers ----

def build_model(torch, width, layers, device, dtype):
    torch.manual_seed(0)  # the same weights for every method
    modules = []
    for i in range(layers):
        modules.append(torch.nn.Linear(width, width, dtype=dtype))
        if i < layers - 1:
            modules.append(torch.nn.GELU())
    return torch.nn.Sequential(*modules).to(device).train()


class Data:
    """A fresh batch per step, seeded by the step number: X ~ N(0, 1), target = X @ U @ V (a fixed rank-64 teacher)."""

    def __init__(self, torch, rows, width, device):
        self.torch, self.rows, self.width, self.device = torch, rows, width, device
        g = torch.Generator().manual_seed(7)  # on the host, then moved: identical on every device
        self.U = (torch.randn(width, RANK, generator=g) / width**0.5).to(device)
        self.V = (torch.randn(RANK, width, generator=g) / RANK**0.5).to(device)

    def batch(self, step):
        torch = self.torch
        try:
            g = torch.Generator(device=self.device).manual_seed(1000 + step)
            x = torch.randn(self.rows, self.width, generator=g, device=self.device)
        except Exception:  # no device generator: make it on the host
            x = torch.randn(self.rows, self.width, generator=torch.Generator().manual_seed(1000 + step)).to(self.device)
        return x.to(torch.bfloat16), ((x @ self.U) @ self.V).to(torch.bfloat16)


def mse(torch, out, target):
    return (out.float() - target.float()).square().mean()


def make_optimizer(torch, model, method, lr=LR):
    """Baselines: torch AdamW on everything. FORGE: every nn.Linear a FusedLinear, the biases on fused AdamW."""
    from forge_tpu import FusedLinear

    if method == "vanilla AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    if method == "fused AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
    state_mode = {"FORGE (bf16)": "bf16", "FORGE (int8)": "int8"}[method]
    for i, child in enumerate(list(model)):
        if isinstance(child, torch.nn.Linear):
            fused = FusedLinear.from_linear(child, state_mode=state_mode, lr=lr)
            fused.fusedStateObject.ensure_buffers(fused.weight)  # m, v made now, eagerly
            model[i] = fused
    fused_weights = {id(m.weight) for m in model.modules() if isinstance(m, FusedLinear)}
    return torch.optim.AdamW([p for p in model.parameters() if id(p) not in fused_weights], lr=lr, fused=True)


def optimizer_states(torch, model, optimizer):
    from forge_tpu import FusedLinear
    for state in optimizer.state.values():
        yield from (t for t in state.values() if isinstance(t, torch.Tensor))
    for module in model.modules():
        if isinstance(module, FusedLinear) and module.fusedStateObject.buffers:
            yield from (t for arrays in module.fusedStateObject.buffers.values() for t in arrays.values())


def compile_step(torch, model, optimizer, method, backend):
    """torch.compile for the model and optimizer.step(). FORGE: FusedLinear.forward is not traced (see the docstring)."""
    from forge_tpu import FusedLinear
    forge = method.startswith("FORGE")
    if forge and not getattr(FusedLinear.forward, "_forge_not_traced", False):
        FusedLinear.forward = torch.compiler.disable(FusedLinear.forward)
        FusedLinear.forward._forge_not_traced = True
    # baselines: fullgraph, so a part that cannot be traced is an error and never a silent eager fallback
    return torch.compile(model, backend=backend, fullgraph=not forge), torch.compile(optimizer.step, backend=backend)


def predict(method, layers, width, rows):
    """The memory, by hand (GiB). See the docstring."""
    W = layers * (width * width + width) * 2
    bias = layers * width * 2
    states = (layers * width * width * 2 * (1 + 4 / 64) + 2 * bias) if method == "FORGE (int8)" else 2 * W
    dw = bias if method.startswith("FORGE") else W
    acts = (2 * layers - 1) * rows * width * 2 + rows * width * 4
    data = 2 * rows * width * 2
    held = W + states + data
    start_of_backward = held + acts + rows * width * 6  # + the loss gradient, fp32 and bf16
    end_of_backward = held + dw
    peak = None if method == "vanilla AdamW" else max(start_of_backward, end_of_backward)  # vanilla: + foreach temporaries
    return {k: (v / GIB if v is not None else None) for k, v in
            {"weights": W, "states": states, "dw": dw, "acts": acts, "held": held, "peak": peak}.items()}


class Memory:
    """The TPU's memory counters, in GiB (all 0 on a CPU, which has none)."""

    def __init__(self, torch, on_tpu):
        self.torch, self.on_tpu = torch, on_tpu

    def sync(self):
        if self.on_tpu:
            self.torch.accelerator.synchronize()

    def reset_peak(self):
        if self.on_tpu:
            self.torch.accelerator.reset_peak_memory_stats()

    def peak(self):
        return self.torch.accelerator.max_memory_allocated() / GIB if self.on_tpu else 0.0

    def held(self):
        return self.torch.accelerator.memory_allocated() / GIB if self.on_tpu else 0.0


def device_and_backend(torch):
    on_tpu = torch.accelerator.is_available() and torch.accelerator.current_accelerator().type == "tpu"
    return on_tpu, ("tpu", "tpu") if on_tpu else ("cpu", "aot_eager")  # cpu: only to check that the script runs


# ---- 2. Correctness: every method against torch AdamW in fp32 ----

def check(k):
    import torch
    from forge_tpu import FusedLinear
    on_tpu, (device, backend) = device_and_backend(torch)
    width, layers, rows, steps = 1024, 4, 256, 10
    say(f"[check] {layers} layers of {width}, {rows} rows, {steps} steps, lr 1e-3, {'compiled' if k['compile'] else 'eager'}, "
        f"against torch AdamW in fp32 ...")
    data = Data(torch, rows, width, device)
    batches = [data.batch(s) for s in range(steps)]

    def train(method, dtype):
        model = build_model(torch, width, layers, device, torch.bfloat16).to(dtype)  # the reference: the SAME bf16 weights, in fp32
        start = [m.weight.detach().float().clone() for m in model if isinstance(m, torch.nn.Linear)]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) if method == "reference" else make_optimizer(torch, model, method, lr=1e-3)
        forward, step = model, optimizer.step
        if k["compile"] and method != "reference":
            forward, step = compile_step(torch, model, optimizer, method, backend)
        losses = []
        for x, target in batches:
            loss = mse(torch, forward(x.to(dtype)), target)
            loss.backward()
            step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())
        weights = [m.weight.detach().float() for m in model if isinstance(m, (torch.nn.Linear, FusedLinear))]
        steps_taken = [m.fusedStateObject.step for m in model if isinstance(m, FusedLinear)]
        return start, weights, losses, steps_taken

    start, reference, reference_losses, _ = train("reference", torch.float32)
    moved = max((r - s).abs().max().item() for r, s in zip(reference, start))
    rows_out = [{"method": "torch AdamW, fp32 (reference)", "losses": reference_losses, "error": 0.0, "moved": moved}]
    for method in k["methods"]:
        torch._dynamo.reset()
        _, weights, losses, steps_taken = train(method, torch.bfloat16)
        error = max((w - r).abs().max().item() for w, r in zip(weights, reference))
        rows_out.append({"method": method, "losses": losses, "error": error, "relative": error / moved,
                         "forge_steps": steps_taken[:1]})
        say(f"[check] {method:<14} max |w - w_ref| = {error:.2e} = {error / moved:.3f} x the reference's largest move   "
            f"loss {losses[0]:.4f} -> {losses[-1]:.4f} (reference {reference_losses[-1]:.4f})")
    return {"width": width, "layers": layers, "rows": rows, "steps": steps, "compiled": k["compile"], "rows_out": rows_out}


# ---- 3. One method at one batch size (its own process) ----

def run(method, rows, k):
    import torch  # only in here: the process that touches the TPU owns it, the parent must not
    from forge_tpu import FusedLinear

    on_tpu, (device, backend) = device_and_backend(torch)
    tag = f"[{method} | batch {rows}]"
    memory = Memory(torch, on_tpu)
    started = time.perf_counter()
    model = build_model(torch, k["width"], k["layers"], device, torch.bfloat16)
    optimizer = make_optimizer(torch, model, method)
    params = sum(p.numel() for p in model.parameters())
    fused_layers = sum(isinstance(m, FusedLinear) for m in model.modules())
    say(f"{tag} MLP {k['layers']} x {k['width']}: {params:,} parameters on {device}, {fused_layers} FusedLinear layers "
        f"({time.perf_counter() - started:.0f} s)")
    data = Data(torch, rows, k["width"], device)

    forward_model, step_optimizer, optimizer_compiled = model, optimizer.step, False
    if k["compile"]:
        forward_model, step_optimizer = compile_step(torch, model, optimizer, method, backend)
        optimizer_compiled = True
    watched = [m.weight for m in model if isinstance(m, (torch.nn.Linear, FusedLinear))]
    watched = [watched[0], watched[-1]]
    before = [w.detach().to("cpu", copy=True).float() for w in watched]

    steps = k["warmup"] + k["timed"]
    step_ms, issued_ms, step_peak, step_held, losses, dw_gib = [], [], [], [], [], 0.0
    for step in range(steps):
        if step == 0:
            say(f"{tag} step 1 {'compiles the model and the optimizer step' if k['compile'] else 'compiles the eager programs'}"
                f" - this one can take minutes ...")
        x, target = data.batch(step)  # made before the clock starts
        memory.sync()
        memory.reset_peak()
        start = time.perf_counter()
        loss = mse(torch, forward_model(x), target)
        loss.backward()  # FORGE updates its layers in here
        if step == 0:  # the weight-gradient pool: every .grad alive now (an untimed warmup step)
            dw_gib = sum(p.grad.nbytes for p in model.parameters() if p.grad is not None) / GIB
        try:
            step_optimizer()
        except Exception as error:
            if not optimizer_compiled or step:
                raise
            say(f"{tag} optimizer.step() could not be compiled ({type(error).__name__}: {str(error)[:200]}); it runs eager")
            step_optimizer, optimizer_compiled = optimizer.step, False
            step_optimizer()
        optimizer.zero_grad(set_to_none=True)
        issued = time.perf_counter()
        memory.sync()  # wait for the TPU before reading the clock
        end = time.perf_counter()
        step_ms.append(1000 * (end - start))
        issued_ms.append(1000 * (issued - start))
        step_peak.append(memory.peak())
        step_held.append(memory.held())
        losses.append(loss.detach().float().item())  # after the clock stopped
        phase = f"warmup {step + 1}/{k['warmup']}" if step < k["warmup"] else f"timed {step + 1 - k['warmup']}/{k['timed']}"
        say(f"{tag} {phase:<13} {step_ms[-1]:9.1f} ms (host {issued_ms[-1]:7.1f})   loss {losses[-1]:8.5f}   "
            f"peak {step_peak[-1]:6.2f} GiB   held {step_held[-1]:6.2f} GiB")

    moved = [(w.detach().to("cpu").float() - b).abs().max().item() for w, b in zip(watched, before)]
    first = next((m for m in model if isinstance(m, FusedLinear)), None)
    forge_steps = first.fusedStateObject.step if first is not None else None  # FORGE's own count of its updates
    breaks = {}
    if k["compile"]:
        import torch._dynamo.utils
        breaks = {str(reason)[:160]: n for reason, n in torch._dynamo.utils.counters["graph_break"].items()}
    gib = lambda tensors: sum(t.nbytes for t in tensors) / GIB
    result = {"method": method, "rows": rows, "width": k["width"], "layers": k["layers"], "compiled": k["compile"],
              "optimizer_compiled": optimizer_compiled, "chip": chip_name(torch) if on_tpu else "cpu",
              "versions": versions(torch), "params": params, "tokens": rows, "warmup": k["warmup"], "timed": k["timed"],
              "steps": steps, "fused_layers": fused_layers, "peak_gib": max(step_peak[k["warmup"]:]),
              "weights_gib": gib(model.parameters()), "states_gib": gib(optimizer_states(torch, model, optimizer)),
              "dw_gib": dw_gib, "warmup_ms": step_ms[:k["warmup"]], "step_ms": step_ms[k["warmup"]:],
              "issued_ms": issued_ms[k["warmup"]:], "step_peak_gib": step_peak, "step_held_gib": step_held,
              "losses": losses, "weights_moved": moved, "graph_breaks": breaks, "forge_steps": forge_steps}

    say(f"{tag} one more step, eager, followed layer by layer (memory figure) ...")
    try:
        x, target = data.batch(steps)
        result["inside_a_step"] = inside_a_step(torch, model, optimizer, lambda: mse(torch, model(x), target), memory)
    except Exception as error:
        result["inside_a_step"], result["inside_error"] = [], f"{type(error).__name__}: {str(error)[:300]}"
        say(f"{tag} the layer-by-layer step failed ({result['inside_error']}); the timed results are kept")
    say(f"{tag} done in {time.perf_counter() - started:.0f} s")
    return result


def inside_a_step(torch, model, optimizer, forward, memory):
    """One eager step, cut after every layer forward and backward: the memory held at each cut, the highest point since
    the cut before, and the weight gradients alive."""
    from forge_tpu import FusedLinear
    points = []
    layers = [m for m in model if isinstance(m, (torch.nn.Linear, FusedLinear))]

    def cut(name):
        memory.sync()
        gradients = sum(p.grad.nbytes for p in model.parameters() if p.grad is not None) / GIB
        points.append({"name": name, "peak": memory.peak(), "held": memory.held(), "dw": gradients})
        memory.reset_peak()

    def before_layer(i):
        def hook(module, args):  # backward reaches this layer's input once the layer's backward is done
            if args[0].requires_grad:
                args[0].register_hook(lambda grad: cut(f"backward, layer {i + 1}"))
        return hook

    handles = [layer.register_forward_pre_hook(before_layer(i)) for i, layer in enumerate(layers)]
    handles += [layer.register_forward_hook(lambda module, args, out, i=i: cut(f"forward, layer {i + 1}")) for i, layer in enumerate(layers)]
    try:
        memory.sync()
        memory.reset_peak()
        cut("start")
        loss = forward()
        cut("forward, loss")
        loss.backward()
        cut("backward, done")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        cut("optimizer step")
    finally:
        for handle in handles:
            handle.remove()
    names = [p["name"] for p in points]
    first_backward = next(i for i, n in enumerate(names) if n.startswith("backward"))
    points[first_backward]["name"] = f"backward, loss to layer {len(layers)}"
    return points


def chip_name(torch):
    for name in (lambda: torch.tpu.get_device_properties().name, lambda: torch.tpu.get_device_name()):
        try:
            found = name()
        except Exception:
            continue
        if found:
            return str(found)
    return "TPU"


def versions(torch):
    import importlib.metadata as md
    found = {"torch": torch.__version__}
    for package in ("torch_tpu", "jax", "jaxlib", "libtpu"):
        try:
            found[package] = md.version(package)
        except Exception:
            pass
    return found


# ---- 4. Everything, each in a fresh process, with live progress ----

def run_everything():
    k = knobs()
    say(f"FORGE-TPU / MLP {k['layers']} x {k['width']}: {'torch.compile' if k['compile'] else 'eager (optimized: defer and fuse)'}, "
        f"batch sizes {k['batch_sizes']}, {k['warmup']} warmup + {k['timed']} timed steps, methods {k['methods']}")
    started = time.perf_counter()
    mode_dir(k).mkdir(parents=True, exist_ok=True)
    if k["check"]:
        found = child(["check"], k, mode_dir(k) / "log check.txt", "correctness check")
        if found:
            (mode_dir(k) / "check.json").write_text(json.dumps(found, indent=1))
            say("\n" + check_table(found) + "\n")
    for rows in k["batch_sizes"]:
        folder = out_dir(k, rows)
        folder.mkdir(parents=True, exist_ok=True)
        results = []
        for method in k["methods"]:
            say(f"\n--- {method} | batch {rows} ---")
            t0 = time.perf_counter()
            result = child(["run", method, str(rows)], k, folder / f"log {method}.txt", f"{method} | batch {rows}")
            result = result or {"method": method, "rows": rows, "error": LAST_ERROR[0]}
            results.append(result)
            (folder / "results.json").write_text(json.dumps(results, indent=1))  # saved after every method
            say(f"  => {method} | batch {rows}: " + (result.get("error") or
                f"{statistics.median(result['step_ms']):.1f} ms, peak {result['peak_gib']:.2f} GiB, loss "
                f"{result['losses'][0]:.4f} -> {result['losses'][-1]:.4f}") + f"  ({time.perf_counter() - t0:.0f} s)")
        report(results, folder, k)
        say(f"=== batch {rows} done; {(time.perf_counter() - started) / 60:.1f} min so far ===")
    summary(k)
    say(f"ALL DONE in {(time.perf_counter() - started) / 60:.1f} min. Results in {k['out']}")


LAST_ERROR = ["crashed"]


def child(arguments, k, log_path, label):
    """Runs this file with the arguments in a fresh process, echoes its progress, returns its RESULT (or None)."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    src = str(Path(__file__).resolve().parent.parent / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if k["compile"]:
        env.pop("TORCH_TPU_DEFER_AND_FUSE", None)
    else:
        env["TORCH_TPU_DEFER_AND_FUSE"] = "1"  # optimized eager, the same for every method
    process = subprocess.Popen([sys.executable, "-u", __file__, *arguments], env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    lines, last = [], {"time": time.perf_counter(), "line": "starting"}
    started = time.perf_counter()

    def read():
        for line in process.stdout:
            lines.append(line)
            if line.startswith("[") and "]" in line[:60] and not line.startswith("[W") or "Error" in line:
                print("  " + line, end="", flush=True)
                last["time"], last["line"] = time.perf_counter(), line.strip()[:90]

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    while reader.is_alive():
        reader.join(timeout=5)
        if reader.is_alive() and time.perf_counter() - last["time"] > 30:
            say(f"  ... still working ({time.perf_counter() - started:.0f} s into {label}; last: {last['line']})")
            last["time"] = time.perf_counter()
    process.wait()
    log = "".join(lines)
    log_path.write_text(log)
    found = [line[len("RESULT "):] for line in lines if line.startswith("RESULT ")]
    if found:
        result = json.loads(found[-1])
        requests = re.findall(r"num_cache_reqs=(\d+)", log)  # TorchTPU prints, at exit, how many programs it fetched
        if requests and "steps" in result:
            result["programs_per_step"] = int(requests[-1]) / (result["steps"] + 1)
        return result
    if "RESOURCE_EXHAUSTED" in log or "out of memory" in log.lower():
        LAST_ERROR[0] = "out of memory"
    else:
        errors = [line.strip() for line in lines if re.match(r"^[\w.]*(Error|Exception|Unsupported)\b.*:", line.strip())]
        cause = [line.strip() for line in lines if "raised exception" in line][-1:]
        tail = errors[-1:] or [line.strip() for line in lines if line.strip()][-1:] or ["crashed"]
        LAST_ERROR[0] = (tail[0] + (f" ({cause[0][-160:]})" if cause else ""))[:400]
    say(f"  {label} failed: {LAST_ERROR[0]}")
    return None


# ---- 5. Tables and figures ----

def check_table(c):
    lines = [f"Correctness: {c['layers']} layers of {c['width']}, {c['rows']} rows, {c['steps']} steps (lr 1e-3), "
             f"{'compiled' if c['compiled'] else 'eager'}; every method in bf16 against torch AdamW in fp32, same weights, same batches.",
             "", "| Method | max \\|w − w_ref\\| | ÷ reference's largest move | loss step 1 → last |", "|---|:--:|:--:|:--:|"]
    for r in c["rows_out"]:
        relative = f"{r['relative']:.3f}" if "relative" in r else "-"
        lines.append(f"| {r['method']} | {r['error']:.2e} | {relative} | {r['losses'][0]:.4f} → {r['losses'][-1]:.4f} |")
    lines += ["", "fused AdamW in bf16 is the yardstick: FORGE (bf16) should be as close to the fp32 reference as it is "
                  "(bf16 rounding of the weights); FORGE (int8) a little further (8-bit states)."]
    return "\n".join(lines)


def summarize(results):
    rows = []
    for r in results:
        if "error" in r:
            rows.append(r)
            continue
        flops = 6 * r["params"] * r["tokens"]  # training: 2 FLOPs per parameter per row forward, 4 backward
        tflops = [flops / (ms / 1000) / 1e12 for ms in r["step_ms"]]
        sd = lambda xs: statistics.stdev(xs) if len(xs) > 1 else 0.0
        trace = r.get("inside_a_step") or []
        top = max(trace, key=lambda p: p["peak"]) if trace else None
        names = {p["name"]: p for p in trace}
        acts = names["forward, loss"]["held"] - names["start"]["held"] if "forward, loss" in names else None
        forge = r["method"].startswith("FORGE")
        rows.append({**r, "updated": all(m > 0 for m in r["weights_moved"]) and (not forge or r["forge_steps"] == r["steps"]),
                     "step": statistics.median(r["step_ms"]), "step_sd": sd(r["step_ms"]),
                     "issued": statistics.median(r["issued_ms"]), "tflops": statistics.mean(tflops),
                     "tflops_sd": sd(tflops), "peak_at": top["name"] if top else "?", "acts_measured": acts,
                     "held_start": names["start"]["held"] if "start" in names else None,
                     "rest_gib": max(r["peak_gib"] - r["weights_gib"] - r["states_gib"] - r["dw_gib"], 0.0),
                     "predicted": predict(r["method"], r["layers"], r["width"], r["rows"])})
    return rows


def report(results, folder, k):
    rows = summarize(results)
    ok = [r for r in rows if "error" not in r]
    first = ok[0] if ok else {}
    chip, size = first.get("chip", "TPU"), (rows[0]["rows"] if rows else "?")
    mode = "torch.compile" if first.get("compiled") else "eager (defer and fuse)"
    lines = [f"Single-chip comparison on {chip}, MLP {k['layers']} x {k['width']} ({first.get('params', 0):,} parameters), "
             f"batch {size} rows, BF16 everywhere, {mode}; step time is the median of {first.get('timed', '?')} steps.", "",
             "| Method | Peak (GiB) | Step (ms) | TF/s | weights updated? | loss step 1 → last |", "|---|:--:|:--:|:--:|:--:|:--:|"]
    for r in rows:
        bold = (lambda s: f"**{s}**") if r["method"].startswith("FORGE") else (lambda s: s)
        if "error" in r:
            lines.append(f"| {bold(r['method'])} | {r['error']} | | | | |")
            continue
        peak = f"{r['peak_gib']:.2f}" + (" (chip full)" if r["peak_gib"] >= CHIP_LIMIT_GIB else "")
        step, tflops = f"{r['step']:.1f} ± {r['step_sd']:.1f}", f"{r['tflops']:.1f} ± {r['tflops_sd']:.1f}"
        updated = "yes" if r["updated"] else "**NO**"
        lines.append(f"| {bold(r['method'])} | {bold(peak)} | {bold(step)} | {bold(tflops)} | {updated} | "
                     f"{r['losses'][0]:.4f} → {r['losses'][-1]:.4f} |")
    lines += ["", "Memory, predicted by hand vs measured (GiB):", "",
              "| Method | weights | opt states | dW pool | held (pred / meas.) | activations (pred / meas.) | peak (pred / meas.) | peak happens at |",
              "|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|"]
    for r in ok:
        p = r["predicted"]
        pair = lambda a, b: f"{a:.2f} / {b:.2f}" if a is not None and b is not None else (f"{a:.2f} / ?" if a is not None else f"? / {b:.2f}" if b is not None else "?")
        lines.append(f"| {r['method']} | {p['weights']:.2f} / {r['weights_gib']:.2f} | {p['states']:.2f} / {r['states_gib']:.2f} "
                     f"| {p['dw']:.2f} / {r['dw_gib']:.2f} | {pair(p['held'], r['held_start'])} | {pair(p['acts'], r['acts_measured'])} "
                     f"| {pair(p['peak'], r['peak_gib'])} | {r['peak_at']} |")
    lines += ["", "held: weights + optimizer states + the batch. Peak prediction: the larger of (held + activations + the "
                  "loss gradient), at the start of backward, and (held + dW pool), at its end. vanilla AdamW is not predicted: "
                  "its optimizer.step() makes temporaries the size of several weight copies."]
    lines += ["", "Diagnostics:", "",
              "| Method | host ms (median) | TPU programs / step (≈) | graph breaks | optimizer.step compiled | FORGE updates / steps |",
              "|---|:--:|:--:|:--:|:--:|:--:|"]
    for r in ok:
        programs = f"{r['programs_per_step']:.0f}" if "programs_per_step" in r else "?"
        breaks = sum(r["graph_breaks"].values()) if r["compiled"] else "-"
        forge = f"{r['forge_steps']} / {r['steps']}" if r["forge_steps"] is not None else "-"
        lines.append(f"| {r['method']} | {r['issued']:.1f} | {programs} | {breaks} | {'yes' if r['optimizer_compiled'] else 'no'} | {forge} |")
    if any(r["compiled"] for r in ok):
        lines.append(f"\nExpected graph breaks: torch.optim's own (1 around optimizer.step(), and fused=True's kernel outside "
                     f"the graph), and for FORGE one per FusedLinear call ({first.get('fused_layers', '?')} here: not traced, "
                     f"they run FORGE's eager code). TorchTPU 0.2.0 also keeps a copy of every weight a compiled forward saves "
                     f"for backward (torch_tpu fx_passes/clone_mutated_returned_placeholders.py): the baselines' compiled peak "
                     f"includes one extra weights-sized copy that FORGE's Linear weights avoid. Compare memory in eager mode.")
    for r in ok:
        if r["graph_breaks"]:
            lines.append(f"\nGraph breaks, {r['method']}: " + "; ".join(f"{n}x {why.splitlines()[0]}" for why, n in r["graph_breaks"].items()))
        if r.get("inside_error"):
            lines.append(f"\nLayer-by-layer step, {r['method']}: failed ({r['inside_error']})")
    if first:
        lines += ["", "Versions: " + ", ".join(f"{p} {v}" for p, v in first["versions"].items())]
    table = "\n".join(lines)
    say("\n" + table + "\n")
    (folder / "table.md").write_text(table + "\n")
    if ok:
        title = f"MLP {k['layers']} x {k['width']}, {chip}: batch {size} rows, BF16, {mode}"
        plot_headline(ok, title, folder)
        plot_memory(ok, title, folder)
        plot_losses(ok, title, folder)
        say(f"table.md, headline.png, memory.png and losses.png saved in {folder}")


# dk4248's colours
WEIGHTS, STATES, ACTIVATIONS, DW = "#2a78d6", "#1baf7a", "#c7ccd6", "#e34948"
POINT = {"vanilla AdamW": "#2a78d6", "fused AdamW": "#8a8880", "FORGE (bf16)": "#13876c", "FORGE (int8)": "#13876c"}
LINE = {"vanilla AdamW": dict(color="#2a78d6", ls="--"), "fused AdamW": dict(color="#8a8880", ls=":"),
        "FORGE (bf16)": dict(color="#13876c", ls="-"), "FORGE (int8)": dict(color="#e34948", ls="-")}
FORGE_SHADE, ARROW, INK, MUTED = "#e9f5ec", "#c62d24", "#1f1f1f", "#6c6a64"


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#d9d8d4", "grid.linestyle": ":", "axes.axisbelow": True})
    return plt


def plot_headline(rows, title, folder):
    """dk4248's Figure 1: (a) the peak and what it holds, (b) memory against speed. A method whose weights were not
    updated is drawn hatched and marked, never left out."""
    plt = pyplot()
    fig, (bars, scatter) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.05, 1]})
    for x, r in enumerate(rows):
        if r["method"].startswith("FORGE"):
            bars.axvspan(x - 0.5, x + 0.5, color=FORGE_SHADE, zorder=0, lw=0)
        bottom = 0.0
        for label, size, color in (("weights", r["weights_gib"], WEIGHTS), ("opt states", r["states_gib"], STATES),
                                   ("activations & temporaries", r["rest_gib"], ACTIVATIONS), ("weight gradient", r["dw_gib"], DW)):
            bars.bar(x, size, 0.62, bottom=bottom, color=color, label=label, zorder=2, hatch=None if r["updated"] else "//")
            bottom += size
        text = f"{r['peak_gib']:.2f}" + ("" if r["updated"] else "\nNOT UPDATED")
        bars.annotate(text, (x, max(bottom, r["peak_gib"])), xytext=(0, 4), textcoords="offset points", ha="center",
                      fontweight="bold", fontsize=12, color=INK if r["updated"] else ARROW)
    bars.set_xticks(range(len(rows)), [r["method"].replace(" ", "\n", 1) for r in rows])
    for tick, r in zip(bars.get_xticklabels(), rows):
        tick.set_fontweight("bold" if r["method"].startswith("FORGE") else "normal")
    by_label = dict(zip(*reversed(bars.get_legend_handles_labels())))
    bars.legend(by_label.values(), by_label.keys(), loc="upper right", framealpha=1)
    bars.set_xlim(-0.6, len(rows) - 0.4)
    bars.set_ylim(0, 1.6 * max(max(r["peak_gib"], r["weights_gib"] + r["states_gib"] + r["dw_gib"]) for r in rows) or 1)
    bars.set_ylabel("Peak memory (GiB)")
    bars.set_title("(a) Peak memory", loc="left", fontweight="bold")
    bars.grid(axis="x", visible=False)

    by_method = {r["method"]: r for r in rows}
    for r in rows:
        forge = r["method"].startswith("FORGE")
        scatter.scatter(r["peak_gib"], r["step"], s=230 if forge else 70, color=POINT[r["method"]], zorder=3,
                        marker="o" if r["updated"] else "X")
        name = r["method"] + ("" if r["updated"] else " (NOT UPDATED)")
        scatter.annotate(name, (r["peak_gib"], r["step"]), xytext=(14, 0) if forge else (-12, -14),
                         textcoords="offset points", ha="left" if forge else "right", va="center", fontsize=13,
                         color=INK if forge else POINT[r["method"]], fontweight="bold" if forge else "normal")
    for start, end in ARROWS:
        if start in by_method and end in by_method and by_method[start]["updated"] and by_method[end]["updated"]:
            a, b = by_method[start], by_method[end]
            scatter.annotate("", (b["peak_gib"], b["step"]), (a["peak_gib"], a["step"]), zorder=2,
                             arrowprops=dict(arrowstyle="-|>", color=ARROW, linestyle="--", lw=2, shrinkA=6, shrinkB=12))
            speedup = a["step"] / b["step"]
            words = f"{speedup:.2f}×  $\\bf{{faster}}$" if speedup >= 1 else f"{1 / speedup:.2f}×  $\\bf{{slower}}$"
            along = lambda key: a[key] + 0.4 * (b[key] - a[key])
            scatter.annotate(words, (along("peak_gib"), along("step")), ha="center", va="center", color=ARROW,
                             fontsize=13, bbox=dict(fc="white", ec="none", pad=1.5), zorder=4)
    scatter.invert_xaxis()
    scatter.invert_yaxis()
    scatter.margins(x=0.75, y=0.3)
    scatter.set_xlabel("Peak memory (GiB)   ← smaller")
    scatter.set_ylabel("Step time (ms)   ↑ faster")
    scatter.set_title("(b) Memory vs. speed", loc="left", fontweight="bold")
    fig.suptitle(title, x=0.01, ha="left", color=MUTED, fontsize=11)
    fig.tight_layout()
    fig.savefig(folder / "headline.png", dpi=200, facecolor="white")
    plt.close(fig)


def plot_memory(rows, title, folder):
    """(c) every step: its own peak, and what it leaves behind. (d) inside one (eager) step, layer by layer."""
    plt = pyplot()
    from matplotlib.lines import Line2D
    fig, (steps, inside) = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1, 1.6]})
    warmup = rows[0]["warmup"]
    for r in rows:
        x = range(1, len(r["step_peak_gib"]) + 1)
        steps.plot(x, r["step_peak_gib"], lw=2, **LINE[r["method"]])
        steps.plot(x, r["step_held_gib"], lw=1.2, alpha=0.7, **{**LINE[r["method"]], "ls": (0, (1, 2))})
    steps.axvspan(0.5, warmup + 0.5, color="#f1f0ec", zorder=0)
    steps.text(warmup / 2 + 0.5, 0.02, "warmup (not timed)", transform=steps.get_xaxis_transform(), ha="center", color=MUTED)
    steps.set_xlabel("Step")
    steps.set_ylabel("Memory (GiB)")
    steps.set_title("(c) Memory at every step", loc="left", fontweight="bold")
    handles = [Line2D([], [], lw=2, label=r["method"], **LINE[r["method"]]) for r in rows]
    handles += [Line2D([], [], color=MUTED, lw=2, label="peak of the step"),
                Line2D([], [], color=MUTED, lw=1.2, ls=(0, (1, 2)), label="held after it")]
    steps.legend(handles=handles, fontsize=9, ncol=3, loc="center right", framealpha=1)

    traced = [r for r in rows if r.get("inside_a_step")]
    if traced:
        names = [p["name"] for p in traced[0]["inside_a_step"]]
        for r in traced:
            xs, ys = [0], [r["inside_a_step"][0]["held"]]
            for i, p in enumerate(r["inside_a_step"][1:], 1):
                xs += [i - 0.5, i]
                ys += [max(p["peak"], ys[-1], p["held"]), p["held"]]
            inside.plot(xs, ys, lw=1.8, label=r["method"], **LINE[r["method"]])
            inside.axhline(r["peak_gib"], lw=0.8, alpha=0.5, **LINE[r["method"]])
        forward_end, backward_end = names.index("forward, loss"), names.index("backward, done")
        for lo, hi, label, shade in ((0, forward_end, "forward", "#f4f7fb"), (forward_end, backward_end, "backward", "#fbf3f3"),
                                     (backward_end, len(names) - 1, "step", "#f3f8f4")):
            inside.axvspan(lo, hi, color=shade, zorder=0)
            inside.text((lo + hi) / 2, 0.02, label, transform=inside.get_xaxis_transform(), ha="center", color=MUTED)
        inside.set_xticks([0, forward_end, backward_end, len(names) - 1], ["start", "loss", "backward done", "step"], fontsize=10)
        inside.legend(loc="upper left", framealpha=1)
    inside.set_ylabel("Memory (GiB)")
    inside.set_title("(d) Inside one step, layer by layer (eager)", loc="left", fontweight="bold")
    inside.text(1, 1.01, "thin lines: the peak of the timed steps", transform=inside.transAxes, ha="right", color=MUTED, fontsize=10)
    fig.suptitle(title, x=0.01, ha="left", color=MUTED, fontsize=11)
    fig.tight_layout()
    fig.savefig(folder / "memory.png", dpi=200, facecolor="white")
    plt.close(fig)


def plot_losses(rows, title, folder):
    """(e) the loss of every step: same weights, same batches, so the curves must lie on top of each other."""
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in rows:
        ax.plot(range(1, len(r["losses"]) + 1), r["losses"], lw=2, label=r["method"] + ("" if r["updated"] else " (NOT UPDATED)"), **LINE[r["method"]])
    ax.axvspan(0.5, rows[0]["warmup"] + 0.5, color="#f1f0ec", zorder=0)
    ax.set_xlabel("Step (a new batch every step)")
    ax.set_ylabel("Training loss (MSE)")
    ax.set_title("(e) Loss at every step", loc="left", fontweight="bold")
    ax.text(1, 1.01, "same weights, same batches: lines that hide each other = the same training", transform=ax.transAxes,
            ha="right", color=MUTED, fontsize=10)
    ax.legend(framealpha=1)
    fig.suptitle(title, x=0.01, ha="left", color=MUTED, fontsize=11)
    fig.tight_layout()
    fig.savefig(folder / "losses.png", dpi=200, facecolor="white")
    plt.close(fig)


def summary(k):
    rows = []
    for size in k["batch_sizes"]:
        path = out_dir(k, size) / "results.json"
        if path.exists():
            rows += summarize(json.loads(path.read_text()))
    if not rows:
        return
    lines = ["| Batch (rows) | Method | Peak (GiB) | predicted | Step (ms) | TF/s | weights updated? |", "|:--:|---|:--:|:--:|:--:|:--:|:--:|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['rows']} | {r['method']} | {r['error']} | | | | |")
            continue
        predicted = f"{r['predicted']['peak']:.2f}" if r["predicted"]["peak"] is not None else "-"
        lines.append(f"| {r['rows']} | {r['method']} | {r['peak_gib']:.2f} | {predicted} | {r['step']:.1f} ± {r['step_sd']:.1f} "
                     f"| {r['tflops']:.1f} | {'yes' if r['updated'] else 'NO'} |")
    (mode_dir(k) / "summary.md").write_text("\n".join(lines) + "\n")
    say("\n" + "\n".join(lines))
    plt = pyplot()
    fig, (peak, step) = plt.subplots(1, 2, figsize=(14, 5))
    for method in ALL_METHODS:
        mine = [r for r in rows if r["method"] == method and "error" not in r]
        if mine:
            sizes = [r["rows"] for r in mine]
            peak.plot(sizes, [r["peak_gib"] for r in mine], marker="o", lw=2, label=method, **LINE[method])
            if mine[0]["predicted"]["peak"] is not None:
                peak.plot(sizes, [r["predicted"]["peak"] for r in mine], lw=1, alpha=0.6, marker="x", **{**LINE[method], "ls": (0, (1, 2))})
            step.plot(sizes, [r["step"] for r in mine], marker="o", lw=2, label=method, **LINE[method])
    for ax, label in ((peak, "Peak memory (GiB)"), (step, "Step time (ms)")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(k["batch_sizes"], [str(s) for s in k["batch_sizes"]])
        ax.set_xlabel("Batch (rows)")
        ax.set_ylabel(label)
        ax.legend(framealpha=1)
    peak.set_title("(f) Peak memory vs. batch (dotted: predicted)", loc="left", fontweight="bold")
    step.set_title("(g) Step time vs. batch", loc="left", fontweight="bold")
    fig.suptitle(f"MLP {k['layers']} x {k['width']}, BF16, {'torch.compile' if k['compile'] else 'eager'}", x=0.01, ha="left", color=MUTED, fontsize=11)
    fig.tight_layout()
    fig.savefig(mode_dir(k) / "scaling.png", dpi=200, facecolor="white")
    plt.close(fig)
    say(f"summary.md and scaling.png saved in {mode_dir(k)}")


if __name__ == "__main__":
    if sys.argv[1:] == ["plot"]:
        k = knobs()
        if (mode_dir(k) / "check.json").exists():
            say(check_table(json.loads((mode_dir(k) / "check.json").read_text())))
        for size in k["batch_sizes"]:
            path = out_dir(k, size) / "results.json"
            if path.exists():
                report(json.loads(path.read_text()), out_dir(k, size), k)
        summary(k)
    elif sys.argv[1:] == ["check"]:
        print("RESULT " + json.dumps(check(knobs())), flush=True)
    elif len(sys.argv) == 4 and sys.argv[1] == "run":
        print("RESULT " + json.dumps(run(sys.argv[2], int(sys.argv[3]), knobs())), flush=True)
    else:
        run_everything()
