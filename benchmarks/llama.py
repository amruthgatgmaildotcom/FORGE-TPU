"""dk4248's "Key results" (peak memory, step time, TF/s, and his two figures) for Llama-3.2-1B on one TPU v5e, at several
sequence lengths, eager or torch.compile.

The model: Llama-3.2-1B's exact architecture (16 layers, hidden 2048, MLP 8192, 32 query / 8 key-value heads of 64,
vocabulary 128,256, lm_head tied to the embedding), with RANDOM weights, built from the config below: nothing is downloaded.
Peak memory, step time and TF/s do not depend on the weight values. Every method starts from the same seeded weights and
trains on the same seeded random tokens (batch 1, labels = input), so the losses of the four methods must match closely.

The methods, all BF16 everywhere (weights AND optimizer states, so FORGE's bf16 states are matched exactly):
    vanilla AdamW   torch.optim.AdamW, its default implementation            (states m, v in the weights' dtype: bf16)
    fused AdamW     torch.optim.AdamW(fused=True)                             (bf16 states)
    FORGE (bf16)    every nn.Linear a FusedLinear (bf16 states), updated inside backward. lm_head is not: it is the
                    embedding too (tied), so it gets two gradients per step. The embedding and the norms: fused AdamW
    FORGE (int8)    the same with int8 states (+ one fp32 scale per 64 rows)

How a step is timed, the same for every method: the clock starts before the forward and stops after
torch.accelerator.synchronize(), which waits for the TPU, following forward + loss, backward (FORGE updates in here),
optimizer.step() and optimizer.zero_grad(). Warmup steps (the first ones compile) are not timed. Step time is the median
of the timed steps (± sd), TF/s is 6 x parameters x tokens / step time step by step (mean ± sd), peak memory is the most
the TPU held during any timed step. "host ms" is when Python had handed the whole step to the TPU (before the sync).

COMPILE_TORCH=1: the model (forward + its backward) and optimizer.step() are each compiled with torch.compile(backend="tpu"),
for every method. The model with fullgraph=True: anything that cannot be traced is an error, never a silent eager fallback.
"FORGE steps eager / compiled" in the diagnostics says which path FORGE's update really took (its two step counters). COMPILE_TORCH=0: eager, in TorchTPU's optimized eager mode (TORCH_TPU_DEFER_AND_FUSE=1), as for Qwen; with DEFER_AND_FUSE=0 in plain eager (results in eager_plain/).

Memory, besides the peak: every step's own peak and what it leaves behind, and one more step (not timed) followed
layer by layer, always EAGER (a sync after every layer cannot happen inside a compiled program): it shows where the
peak happens. The bars split the peak into weights, optimizer states, the weight-gradient pool (every .grad alive after
backward) and the rest (activations and temporaries).

    PYTHONPATH=src python benchmarks/llama.py          every method x every sequence length, tables and figures
    PYTHONPATH=src python benchmarks/llama.py plot     only redraws them from the results.json files
Knobs (environment variables): COMPILE_TORCH (1/0), SEQ_LENS (256,512,1024), WARMUP_STEPS (10), TIMED_STEPS (20),
METHODS (";"-separated), OUT_DIR (benchmarks/llama_results), DEFER_AND_FUSE (1/0, eager only), PROFILE_STEPS (0: after the timed steps, this many more
steps are traced with torch_tpu's profiler into "trace <method>/" for xprof). LLAMA_TINY=1: a 2-layer, small-vocabulary model, only to
check that the script runs (on a CPU too).
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
LR = 1e-4  # the other hyperparameters are the same defaults on both sides: betas 0.9 / 0.999, eps 1e-8, weight decay 0.01
BATCH = 1


def knobs():
    env = os.environ.get
    return {"compile": env("COMPILE_TORCH", "1") == "1",
            "seq_lens": [int(s) for s in env("SEQ_LENS", "256,512,1024").split(",") if s.strip()],
            "warmup": int(env("WARMUP_STEPS", "10")), "timed": int(env("TIMED_STEPS", "20")),
            "methods": [m.strip() for m in env("METHODS", ";".join(ALL_METHODS)).split(";") if m.strip()],
            "out": Path(env("OUT_DIR", str(Path(__file__).resolve().parent / "llama_results"))),
            "tiny": env("LLAMA_TINY", "0") == "1", "defer": env("DEFER_AND_FUSE", "1") == "1",
            "profile": int(env("PROFILE_STEPS", "0"))}


def mode_dir(k):
    return "compiled" if k["compile"] else "eager" if k["defer"] else "eager_plain"


def out_dir(k, seq):
    return k["out"] / mode_dir(k) / f"seq_{seq}"


def say(*parts):
    print(*parts, flush=True)


# ---- 1. The model ----

def llama_config(tiny=False):
    """Llama-3.2-1B (meta-llama/Llama-3.2-1B's config.json), written out so that nothing needs downloading."""
    from transformers import LlamaConfig
    kw = dict(vocab_size=128256, hidden_size=2048, intermediate_size=8192, num_hidden_layers=16, num_attention_heads=32,
              num_key_value_heads=8, head_dim=64, hidden_act="silu", max_position_embeddings=131072, rms_norm_eps=1e-5,
              rope_theta=500000.0, tie_word_embeddings=True, attention_bias=False, mlp_bias=False, initializer_range=0.02,
              bos_token_id=128000, eos_token_id=128001, use_cache=False)
    if tiny:
        kw.update(vocab_size=1024, hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=64, bos_token_id=None, eos_token_id=None)
    rope = {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192}
    try:
        return LlamaConfig(**kw, rope_scaling=rope)
    except Exception:  # a transformers version that wants the rope settings another way: they do not change the cost
        return LlamaConfig(**kw)


def build_model(torch, config, device):
    from transformers import LlamaForCausalLM
    torch.manual_seed(0)  # the same weights for every method
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)  # made in bf16 directly: no fp32 copy on the host
    try:
        model = LlamaForCausalLM(config)
    finally:
        torch.set_default_dtype(default)
    model = model.to(device)
    model.tie_weights()  # lm_head and the embedding are one tensor: tie them again in case the move split them
    return model.train()


def make_optimizer(torch, model, method, lr=LR):
    """The baselines update every weight in optimizer.step(). FORGE swaps every nn.Linear (but lm_head) for a FusedLinear,
    which updates its own weight inside backward, and leaves the rest (embedding, norms) to fused AdamW, as the paper does."""
    from forge_tpu import FusedLinear

    if method == "vanilla AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    if method == "fused AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
    state_mode = {"FORGE (bf16)": "bf16", "FORGE (int8)": "int8"}[method]
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, torch.nn.Linear) and child is not model.lm_head:  # lm_head is the embedding too (tied)
                fused = FusedLinear.from_linear(child, state_mode=state_mode, lr=lr)
                fused.fusedStateObject.ensure_buffers(fused.weight)  # m, v made now, eagerly, before anything is compiled
                setattr(module, name, fused)
    fused_weights = {id(m.weight) for m in model.modules() if isinstance(m, FusedLinear)}
    return torch.optim.AdamW([p for p in model.parameters() if id(p) not in fused_weights], lr=lr, fused=True)


def optimizer_states(torch, model, optimizer):
    from forge_tpu import FusedLinear
    for state in optimizer.state.values():
        yield from (t for t in state.values() if isinstance(t, torch.Tensor))
    for module in model.modules():
        if isinstance(module, FusedLinear) and module.fusedStateObject.buffers:
            yield from (t for arrays in module.fusedStateObject.buffers.values() for t in arrays.values())


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
        return self.torch.accelerator.max_memory_allocated() / 2**30 if self.on_tpu else 0.0

    def held(self):
        return self.torch.accelerator.memory_allocated() / 2**30 if self.on_tpu else 0.0


# ---- 2. One method at one sequence length (its own process) ----

def run(method, seq, k):
    import torch  # only in here: the process that imports torch and touches the TPU owns it, the parent must not
    from forge_tpu import FusedLinear

    on_tpu = torch.accelerator.is_available() and torch.accelerator.current_accelerator().type == "tpu"
    device, backend = ("tpu", "tpu") if on_tpu else ("cpu", "aot_eager")  # cpu: only to check that the script runs
    tag = f"[{method} | seq {seq}]"
    memory = Memory(torch, on_tpu)
    started = time.perf_counter()
    say(f"{tag} building Llama-3.2-1B{' (TINY)' if k['tiny'] else ''} with random weights (seed 0) on the host, then moving it to {device} ...")
    model = build_model(torch, llama_config(k["tiny"]), device)
    optimizer = make_optimizer(torch, model, method)
    params = sum(p.numel() for p in model.parameters())
    say(f"{tag} {params / 1e9:.3f} B parameters on {device} after {time.perf_counter() - started:.0f} s; "
        f"{sum(isinstance(m, FusedLinear) for m in model.modules())} FusedLinear layers")

    tokens = torch.randint(model.config.vocab_size, (BATCH, seq), generator=torch.Generator().manual_seed(0)).to(device)
    forward_model, step_optimizer, optimizer_compiled = model, optimizer.step, False
    if k["compile"]:
        import torch._dynamo
        # forward, and its backward (AOTAutograd), as compiled programs. fullgraph: if anything (FORGE included) cannot be
        # traced, fail loudly instead of quietly running that part eager under a "compiled" label
        forward_model = torch.compile(model, backend=backend, fullgraph=True)
        step_optimizer = torch.compile(optimizer.step, backend=backend)
        optimizer_compiled = True
    loss_of = lambda m: m(input_ids=tokens, labels=tokens, use_cache=False).loss  # labels = input: the next-token loss

    watched = [model.model.layers[0].self_attn.q_proj.weight, model.model.layers[-1].mlp.down_proj.weight]
    before = [w.detach().to("cpu", copy=True).float() for w in watched]

    steps = k["warmup"] + k["timed"]
    step_ms, issued_ms, step_peak, step_held, losses, dw_gib = [], [], [], [], [], 0.0
    for step in range(steps):
        if step == 0:
            say(f"{tag} step 1 {'compiles the model and the optimizer step' if k['compile'] else 'compiles the eager programs'}"
                f" - this one can take minutes ...")
        memory.sync()
        memory.reset_peak()
        start = time.perf_counter()
        loss = loss_of(forward_model)
        loss.backward()  # FORGE updates its layers in here
        if step == 0:  # the weight-gradient pool: every .grad alive now, before the optimizer (an untimed warmup step)
            dw_gib = sum(p.grad.nbytes for p in model.parameters() if p.grad is not None) / 2**30
        try:
            step_optimizer()  # the baselines update every weight here, FORGE the embedding and the norms
        except Exception as error:
            if not optimizer_compiled or step:
                raise
            say(f"{tag} optimizer.step() could not be compiled ({type(error).__name__}: {str(error)[:200]}); it runs eager")
            step_optimizer, optimizer_compiled = optimizer.step, False
            step_optimizer()
        optimizer.zero_grad(set_to_none=True)
        issued = time.perf_counter()
        memory.sync()  # the TPU runs ahead of Python: wait for it before reading the clock
        end = time.perf_counter()
        step_ms.append(1000 * (end - start))
        issued_ms.append(1000 * (issued - start))
        step_peak.append(memory.peak())
        step_held.append(memory.held())
        losses.append(loss.detach().float().item())  # after the clock stopped
        phase = f"warmup {step + 1}/{k['warmup']}" if step < k["warmup"] else f"timed {step + 1 - k['warmup']}/{k['timed']}"
        say(f"{tag} {phase:<13} {step_ms[-1]:9.1f} ms (host {issued_ms[-1]:7.1f})   loss {losses[-1]:7.4f}   "
            f"peak {step_peak[-1]:6.2f} GiB   held {step_held[-1]:6.2f} GiB")

    if k["profile"] and on_tpu:  # extra steps after the timed ones, traced (TPU + host) for xprof: they change no timing
        from torch_tpu._internal import profiler
        trace = out_dir(k, seq) / f"trace {method}"
        say(f"{tag} tracing {k['profile']} more steps into {trace} ...")
        try:
            with profiler.profile(activities=[profiler.ProfilerActivity.TPU, profiler.ProfilerActivity.CPU],
                                  on_trace_ready=profiler.xprof_trace_handler(dir_name=str(trace))):
                for _ in range(k["profile"]):
                    loss_of(forward_model).backward()
                    step_optimizer()
                    optimizer.zero_grad(set_to_none=True)
                    memory.sync()
        except Exception as error:  # a failed trace must not lose the timed results
            say(f"{tag} tracing failed ({type(error).__name__}: {str(error)[:200]}); the timed results are kept")
    moved = [(w.detach().to("cpu").float() - b).abs().max().item() for w, b in zip(watched, before)]
    fused = model.model.layers[0].self_attn.q_proj
    forge_steps = None  # which counter FORGE incremented: the Python int (eager path) or step_tensor (compiled path)
    if isinstance(fused, FusedLinear):
        forge_steps = [fused.fusedStateObject.step, int(fused.fusedStateObject.step_tensor.item())]
    breaks = {}
    if k["compile"]:
        import torch._dynamo.utils
        breaks = {str(reason)[:160]: n for reason, n in torch._dynamo.utils.counters["graph_break"].items()}
    gib = lambda tensors: sum(t.nbytes for t in tensors) / 2**30
    result = {"method": method, "seq": seq, "compiled": k["compile"], "optimizer_compiled": optimizer_compiled,
              "chip": chip_name(torch) if on_tpu else "cpu", "versions": versions(torch), "params": params,
              "tokens": BATCH * seq, "warmup": k["warmup"], "timed": k["timed"],
              "peak_gib": max(step_peak[k["warmup"]:]), "weights_gib": gib(model.parameters()),
              "states_gib": gib(optimizer_states(torch, model, optimizer)), "dw_gib": dw_gib,
              "warmup_ms": step_ms[:k["warmup"]], "step_ms": step_ms[k["warmup"]:], "issued_ms": issued_ms[k["warmup"]:],
              "step_peak_gib": step_peak, "step_held_gib": step_held, "losses": losses, "weights_moved": moved,
              "graph_breaks": breaks, "forge_steps": forge_steps, "defer_and_fuse": os.environ.get("TORCH_TPU_DEFER_AND_FUSE") == "1"}

    # one more step, eager, cut layer by layer (with a compiled run, FORGE's step count lives on the device: copy it over)
    say(f"{tag} one more step, eager, followed layer by layer (memory figure) ...")
    try:
        for m in model.modules():
            if isinstance(m, FusedLinear) and m.fusedStateObject.step_tensor is not None and k["compile"]:
                m.fusedStateObject.step = int(m.fusedStateObject.step_tensor.item())
        result["inside_a_step"] = inside_a_step(model, optimizer, lambda: loss_of(model), memory)
    except Exception as error:
        result["inside_a_step"], result["inside_error"] = [], f"{type(error).__name__}: {str(error)[:300]}"
        say(f"{tag} the layer-by-layer step failed ({result['inside_error']}); the timed results are kept")
    say(f"{tag} done in {time.perf_counter() - started:.0f} s")
    return result


def inside_a_step(model, optimizer, forward, memory):
    """One eager step, cut after every layer forward and backward: at each cut the memory held, the highest point since the
    cut before, and the weight gradients alive. Every cut syncs, so this step runs in small pieces: it shows where the peak
    is, the timed steps say how high."""
    points = []

    def cut(name):
        memory.sync()
        gradients = sum(p.grad.nbytes for p in model.parameters() if p.grad is not None) / 2**30
        points.append({"name": name, "peak": memory.peak(), "held": memory.held(), "dw": gradients})
        memory.reset_peak()

    def before_layer(i):
        def hook(module, args, kwargs):  # backward reaches this layer's input once the layer's backward is done
            hidden = args[0] if args else kwargs["hidden_states"]
            if hidden.requires_grad:
                hidden.register_hook(lambda grad: cut(f"backward, layer {i + 1}"))
        return hook

    layers = model.model.layers
    handles = [layer.register_forward_pre_hook(before_layer(i), with_kwargs=True) for i, layer in enumerate(layers)]
    handles += [layer.register_forward_hook(lambda module, args, out, i=i: cut(f"forward, layer {i + 1}")) for i, layer in enumerate(layers)]
    try:
        memory.sync()
        memory.reset_peak()
        cut("start")
        loss = forward()
        cut("forward, loss")
        loss.backward()
        cut("backward, embedding")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        cut("optimizer step")
    finally:
        for handle in handles:
            handle.remove()
    names = [p["name"] for p in points]  # the first backward cut covers the loss, lm_head and the last layer
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
    for package in ("torch_tpu", "jax", "jaxlib", "libtpu", "transformers"):
        try:
            found[package] = md.version(package)
        except Exception:
            pass
    return found


# ---- 3. Every method x every sequence length, each in a fresh process, with live progress ----

def run_everything():
    k = knobs()
    say(f"FORGE-TPU / Llama-3.2-1B: {'torch.compile' if k['compile'] else 'eager (optimized: defer and fuse)' if k['defer'] else 'eager (plain)'}, "
        f"sequence lengths {k['seq_lens']}, {k['warmup']} warmup + {k['timed']} timed steps, methods {k['methods']}")
    everything_started = time.perf_counter()
    for seq in k["seq_lens"]:
        folder = out_dir(k, seq)
        folder.mkdir(parents=True, exist_ok=True)
        results = []
        for method in k["methods"]:
            results.append(run_one(method, seq, k, folder))
            (folder / "results.json").write_text(json.dumps(results, indent=1))  # saved after every method
        report(results, folder)
        say(f"=== sequence {seq} done; {(time.perf_counter() - everything_started) / 60:.1f} min so far ===")
    summary(k)
    say(f"ALL DONE in {(time.perf_counter() - everything_started) / 60:.1f} min. Results in {k['out']}")


def run_one(method, seq, k, folder):
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    src = str(Path(__file__).resolve().parent.parent / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if not k["compile"] and k["defer"]:
        env["TORCH_TPU_DEFER_AND_FUSE"] = "1"  # optimized eager, the same for every method (as for Qwen)
    else:
        env.pop("TORCH_TPU_DEFER_AND_FUSE", None)
    say(f"\n--- {method} | sequence {seq} ---")
    child = subprocess.Popen([sys.executable, "-u", __file__, "run", method, str(seq)], env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    lines, last = [], {"time": time.perf_counter(), "line": "starting"}
    started = time.perf_counter()

    def read():
        for line in child.stdout:
            lines.append(line)
            if line.startswith("[") and "]" in line[:60] or "Error" in line or "error:" in line:
                print("  " + line, end="", flush=True)
                last["time"], last["line"] = time.perf_counter(), line.strip()[:90]

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    while reader.is_alive():
        reader.join(timeout=5)
        if reader.is_alive() and time.perf_counter() - last["time"] > 30:  # nothing for 30 s: say we are still here
            say(f"  ... still working ({time.perf_counter() - started:.0f} s into {method} | seq {seq}; last: {last['line']})")
            last["time"] = time.perf_counter()
    child.wait()
    log = "".join(lines)
    (folder / f"log {method}.txt").write_text(log)  # everything the run printed
    found = [line[len("RESULT "):] for line in lines if line.startswith("RESULT ")]
    if found:
        result = json.loads(found[-1])
    elif "RESOURCE_EXHAUSTED" in log or "out of memory" in log.lower():
        result = {"method": method, "seq": seq, "error": "out of memory"}
    else:
        errors = [line.strip() for line in lines if re.match(r"^[\w.]*(Error|Exception|Unsupported)\b.*:", line.strip())]
        tail = errors[-1:] or [line.strip() for line in lines if line.strip()][-1:] or ["crashed"]
        cause = [line.strip() for line in lines if "raised exception" in line][-1:]  # Dynamo's: the exception inside the trace
        result = {"method": method, "seq": seq, "error": (tail[0] + (f" ({cause[0][-160:]})" if cause else ""))[:400]}
    requests = re.findall(r"num_cache_reqs=(\d+)", log)  # TorchTPU prints, at exit, how many compiled programs it fetched
    if requests and "error" not in result:
        result["programs_per_step"] = int(requests[-1]) / (result["warmup"] + result["timed"] + 1)
    say(f"  => {method} | seq {seq}: " + (result.get("error") or
        f"{statistics.median(result['step_ms']):.1f} ms, peak {result['peak_gib']:.2f} GiB, loss "
        f"{result['losses'][0]:.3f} -> {result['losses'][-1]:.3f}") + f"  ({time.perf_counter() - started:.0f} s)")
    return result


# ---- 4. Tables and figures ----

def summarize(results):
    rows = []
    for r in results:
        if "error" in r:
            rows.append(r)
            continue
        flops = 6 * r["params"] * r["tokens"]  # training: 2 FLOPs per parameter per token forward, 4 backward
        tflops = [flops / (ms / 1000) / 1e12 for ms in r["step_ms"]]
        sd = lambda xs: statistics.stdev(xs) if len(xs) > 1 else 0.0
        top = max(r["inside_a_step"], key=lambda p: p["peak"]) if r.get("inside_a_step") else None
        rows.append({**r, "trains": all(m > 0 for m in r["weights_moved"]), "step": statistics.median(r["step_ms"]),
                     "step_sd": sd(r["step_ms"]), "issued": statistics.median(r["issued_ms"]),
                     "tflops": statistics.mean(tflops), "tflops_sd": sd(tflops), "peak_at": top["name"] if top else "?",
                     "rest_gib": max(r["peak_gib"] - r["weights_gib"] - r["states_gib"] - r["dw_gib"], 0.0)})
    return rows


def report(results, folder):
    rows = summarize(results)
    ok = [r for r in rows if "error" not in r]
    first = ok[0] if ok else {}
    chip, seq = first.get("chip", "TPU"), (rows[0]["seq"] if rows else "?")
    mode = "torch.compile" if first.get("compiled") else "eager (defer and fuse)" if first.get("defer_and_fuse", True) else "eager (plain)"
    lines = [f"Single-chip comparison on {chip}, Llama-3.2-1B (random weights), batch {BATCH}, sequence {seq}, BF16 "
             f"everywhere, {mode}; step time is the median of {first.get('timed', '?')} steps.", "",
             "| Method | Peak (GiB) | Step (ms) | TF/s | Trains? (loss, step 1 → last) |", "|---|:--:|:--:|:--:|:--:|"]
    for r in rows:
        bold = (lambda s: f"**{s}**") if r["method"].startswith("FORGE") else (lambda s: s)
        if "error" in r:
            lines.append(f"| {bold(r['method'])} | {r['error']} | | | |")
            continue
        check = (f"yes ({r['losses'][0]:.3f} → {r['losses'][-1]:.3f})" if r["trains"] else
                 f"**NO: the watched weights never moved** ({r['losses'][0]:.3f} → {r['losses'][-1]:.3f})")
        peak, step = f"{r['peak_gib']:.2f}", f"{r['step']:.1f} ± {r['step_sd']:.1f}"
        tflops = f"{r['tflops']:.1f} ± {r['tflops_sd']:.1f}"
        lines.append(f"| {bold(r['method'])} | {bold(peak)} | {bold(step)} | {bold(tflops)} | {check} |")
    lines += ["", "Diagnostics:", "",
              "| Method | host ms (median) | TPU programs / step (≈) | graph breaks | optimizer.step compiled | FORGE steps eager / compiled | peak happens at | weights / states / dW pool (GiB) |",
              "|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|"]
    for r in ok:
        programs = f"{r['programs_per_step']:.0f}" if "programs_per_step" in r else "?"
        breaks = sum(r["graph_breaks"].values()) if r["compiled"] else "-"
        forge = " / ".join(map(str, r["forge_steps"])) if r.get("forge_steps") else "-"
        lines.append(f"| {r['method']} | {r['issued']:.1f} | {programs} | {breaks} | {'yes' if r['optimizer_compiled'] else 'no'} "
                     f"| {forge} | {r['peak_at']} | {r['weights_gib']:.2f} / {r['states_gib']:.2f} / {r['dw_gib']:.2f} |")
    if any(r["graph_breaks"] for r in ok):
        lines.append("\nExpected graph breaks, from torch.optim itself: 1-2 user-inserted ones around optimizer.step() "
                     "(no_grad, pytorch/pytorch#104053), and fused=True's kernel runs outside the graph (torch.compiler.disable). "
                     "Any other break means part of the step ran eager.")
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
    good = [r for r in ok if r["trains"]]  # a method that does not train is no result: not drawn
    if good:
        title = f"Llama-3.2-1B (random weights), {chip}: batch {BATCH}, sequence {seq}, BF16, {mode}"
        plot_headline(good, title, folder)
        plot_memory(good, title, folder)
        plot_losses(good, title, folder)
        say(f"table.md, headline.png, memory.png and losses.png saved in {folder}")


# dk4248's colours: the memory parts, the methods' points, the FORGE shading and the red arrows
WEIGHTS, STATES, ACTIVATIONS, DW = "#2a78d6", "#1baf7a", "#c7ccd6", "#e34948"
POINT = {"vanilla AdamW": "#2a78d6", "fused AdamW": "#8a8880", "FORGE (bf16)": "#13876c", "FORGE (int8)": "#13876c"}
LINE = {"vanilla AdamW": dict(color="#2a78d6", ls="--"), "fused AdamW": dict(color="#8a8880", ls=":"),
        "FORGE (bf16)": dict(color="#13876c", ls="-"), "FORGE (int8)": dict(color="#e34948", ls="-")}
FORGE_SHADE, ARROW, INK, MUTED = "#e9f5ec", "#c62d24", "#1f1f1f", "#6c6a64"


def pyplot():
    import matplotlib
    matplotlib.use("Agg")  # files only, no window
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#d9d8d4", "grid.linestyle": ":", "axes.axisbelow": True})
    return plt


def plot_headline(rows, title, folder):
    """dk4248's Figure 1: (a) the peak and what it holds, (b) memory against speed."""
    plt = pyplot()
    fig, (bars, scatter) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.05, 1]})
    for x, r in enumerate(rows):
        if r["method"].startswith("FORGE"):
            bars.axvspan(x - 0.5, x + 0.5, color=FORGE_SHADE, zorder=0, lw=0)
        bottom = 0.0
        for label, size, color in (("weights", r["weights_gib"], WEIGHTS), ("opt states", r["states_gib"], STATES),
                                   ("activations & temporaries", r["rest_gib"], ACTIVATIONS), ("weight gradient", r["dw_gib"], DW)):
            bars.bar(x, size, 0.62, bottom=bottom, color=color, label=label, zorder=2)
            bottom += size
        bars.annotate(f"{r['peak_gib']:.2f}", (x, max(bottom, r["peak_gib"])), xytext=(0, 4), textcoords="offset points",
                      ha="center", fontweight="bold", fontsize=13, color=INK)
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
        scatter.scatter(r["peak_gib"], r["step"], s=230 if forge else 70, color=POINT[r["method"]], zorder=3)
        scatter.annotate(r["method"], (r["peak_gib"], r["step"]), xytext=(14, 0) if forge else (-12, -14),
                         textcoords="offset points", ha="left" if forge else "right", va="center", fontsize=13,
                         color=INK if forge else POINT[r["method"]], fontweight="bold" if forge else "normal")
    for start, end in ARROWS:
        if start in by_method and end in by_method:
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
                ys += [max(p["peak"], ys[-1], p["held"]), p["held"]]  # the highest point between two cuts, then the cut
            inside.plot(xs, ys, lw=1.8, label=r["method"], **LINE[r["method"]])
            inside.axhline(r["peak_gib"], lw=0.8, alpha=0.5, **LINE[r["method"]])
        forward_end = names.index("forward, loss")
        backward_end = names.index("backward, embedding")
        for lo, hi, label, shade in ((0, forward_end, "forward", "#f4f7fb"), (forward_end, backward_end, "backward", "#fbf3f3"),
                                     (backward_end, len(names) - 1, "step", "#f3f8f4")):
            inside.axvspan(lo, hi, color=shade, zorder=0)
            inside.text((lo + hi) / 2, 0.02, label, transform=inside.get_xaxis_transform(), ha="center", color=MUTED)
        middle = f"layer {(sum(n.startswith('forward, layer') for n in names) + 1) // 2}"
        ticks = {0: "start", forward_end: "loss", len(names) - 1: "step"}
        ticks.update({names.index(f"{phase}, {middle}"): middle for phase in ("forward", "backward") if f"{phase}, {middle}" in names})
        inside.set_xticks(list(ticks), list(ticks.values()), fontsize=10)
        inside.legend(loc="upper left", framealpha=1)
    inside.set_ylabel("Memory (GiB)")
    inside.set_title("(d) Inside one step, layer by layer (eager)", loc="left", fontweight="bold")
    inside.text(1, 1.01, "thin lines: the peak of the timed steps", transform=inside.transAxes, ha="right", color=MUTED, fontsize=10)
    fig.suptitle(title, x=0.01, ha="left", color=MUTED, fontsize=11)
    fig.tight_layout()
    fig.savefig(folder / "memory.png", dpi=200, facecolor="white")
    plt.close(fig)


def plot_losses(rows, title, folder):
    """(e) the loss of every step, for every method: they must lie on top of each other (same weights, same tokens)."""
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in rows:
        ax.plot(range(1, len(r["losses"]) + 1), r["losses"], lw=2, label=r["method"], **LINE[r["method"]])
    ax.axvspan(0.5, rows[0]["warmup"] + 0.5, color="#f1f0ec", zorder=0)
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Training loss (log scale)")
    ax.set_title("(e) Loss at every step", loc="left", fontweight="bold")
    ax.text(1, 1.01, "same weights, same tokens: lines that hide each other = the same training", transform=ax.transAxes,
            ha="right", color=MUTED, fontsize=10)
    ax.legend(framealpha=1)
    fig.suptitle(title, x=0.01, ha="left", color=MUTED, fontsize=11)
    fig.tight_layout()
    fig.savefig(folder / "losses.png", dpi=200, facecolor="white")
    plt.close(fig)


def summary(k):
    """Every sequence length in one table and one figure: peak memory and step time against the sequence length."""
    rows = []
    for seq in k["seq_lens"]:
        path = out_dir(k, seq) / "results.json"
        if path.exists():
            rows += summarize(json.loads(path.read_text()))
    if not rows:
        return
    lines = ["| Sequence | Method | Peak (GiB) | Step (ms) | TF/s | loss step 1 → last |", "|:--:|---|:--:|:--:|:--:|:--:|"]
    for r in rows:
        lines.append(f"| {r['seq']} | {r['method']} | " + (f"{r['error']} | | | |" if "error" in r else
                     f"{r['peak_gib']:.2f} | {r['step']:.1f} ± {r['step_sd']:.1f} | {r['tflops']:.1f} | "
                     f"{r['losses'][0]:.3f} → {r['losses'][-1]:.3f} |"))
    folder = k["out"] / mode_dir(k)
    (folder / "summary.md").write_text("\n".join(lines) + "\n")
    say("\n" + "\n".join(lines))
    plt = pyplot()
    fig, (peak, step) = plt.subplots(1, 2, figsize=(14, 5))
    for method in ALL_METHODS:
        mine = [r for r in rows if r["method"] == method and "error" not in r]
        if mine:
            seqs = [r["seq"] for r in mine]
            peak.plot(seqs, [r["peak_gib"] for r in mine], marker="o", lw=2, label=method, **LINE[method])
            step.plot(seqs, [r["step"] for r in mine], marker="o", lw=2, label=method, **LINE[method])
    for ax, label in ((peak, "Peak memory (GiB)"), (step, "Step time (ms)")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(k["seq_lens"], [str(s) for s in k["seq_lens"]])
        ax.set_xlabel("Sequence length (tokens, batch 1)")
        ax.set_ylabel(label)
        ax.legend(framealpha=1)
    peak.set_title("(f) Peak memory vs. sequence length", loc="left", fontweight="bold")
    step.set_title("(g) Step time vs. sequence length", loc="left", fontweight="bold")
    fig.suptitle(f"Llama-3.2-1B (random weights), BF16, {'torch.compile' if k['compile'] else 'eager'}", x=0.01, ha="left", color=MUTED, fontsize=11)
    fig.tight_layout()
    fig.savefig(folder / "scaling.png", dpi=200, facecolor="white")
    plt.close(fig)
    say(f"summary.md and scaling.png saved in {folder}")


if __name__ == "__main__":
    if sys.argv[1:] == ["plot"]:
        k = knobs()
        for seq in k["seq_lens"]:
            path = out_dir(k, seq) / "results.json"
            if path.exists():
                report(json.loads(path.read_text()), out_dir(k, seq))
        summary(k)
    elif len(sys.argv) == 4 and sys.argv[1] == "run":  # started by run_one: one method at one length, printed as RESULT {json}
        print("RESULT " + json.dumps(run(sys.argv[2], int(sys.argv[3]), knobs())), flush=True)
    else:
        run_everything()
