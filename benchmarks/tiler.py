import functools, itertools, json, math, os, sys, time, torch
import jax, jax.numpy as jnp
from forge_tpu import _bridge, kernel
from config import OPTIM, tiler_path

#   python benchmarks/tiler.py <model>  (then  python benchmarks/tiler.py report)  ->  results/tiler/<model>.json, results/tiler/tiles.md
# Which tiles (bv, bh, bbt) are the fastest ? The kernel uses one fixed triple for every layer (kernel/_utils.py DEFAULT_TILES). This file times many triples on
# every fused layer shape of the three benchmark models : only the fused update of ONE layer (random go, x, w), not a training step.
# KERNEL TIME ONLY : one torch op launch costs about 0.2 ms of Python (more than the kernel of a small layer), so a timed loop of launches measures Python, not the kernel.
# Here ONE launch runs the kernel `count` times in a loop inside one TPU program (see repeat). ms of one kernel = (time of 2 x REPS kernels - time of REPS kernels) / REPS :
# the launch, the sync and the read are in both times, the difference keeps only the kernels.
# Each triple is a new XLA compile (the slow part), so there is a time limit : per model MAX_SECONDS, shared equally by its layer shapes. The default triple goes first,
# then the triples closest to it. What was not reached is not in the report. Nothing here changes DEFAULT_TILES : read the report, then change it by hand.
DEVICE, DTYPE = "tpu", torch.bfloat16
MODELS = {"mlp":   (4096, {(16384, 4096): 8, (4096, 16384): 8, (4096, 4096): 1}),                                    # name : (BT, {(V, H) : how many such fused layers})
          "llama": (1024, {(2048, 2048): 32, (512, 2048): 32, (8192, 2048): 32, (2048, 8192): 16}),                 # q o | k v | gate up | down, x 16 layers
          "qwen":  (1024, {(2048, 1024): 28, (1024, 1024): 56, (1024, 2048): 28, (3072, 1024): 56, (1024, 3072): 28}),
          # Llama-3.1-8B split over 8 chips (kaggle/tp.py) : the layer shapes of ONE chip, so one Colab chip times exactly what each Kaggle chip runs. q | k v | o | gate up | down, x 32 layers
          "llama8b_seq512":  (512,  {(512, 4096): 32, (128, 4096): 64, (4096, 512): 32, (1792, 4096): 64, (4096, 1792): 32}),
          "llama8b_seq1024": (1024, {(512, 4096): 32, (128, 4096): 64, (4096, 512): 32, (1792, 4096): 64, (4096, 1792): 32})}   # q | k v | o | gate up | down, x 28 layers
BV, BH, BBT = (256, 512, 1024, 2048), (512, 1024, 2048, 4096), (512, 1024, 2048)  # every combination is tried (cut to the layer : bv <= V, bh <= H, bbt <= BT)
DEFAULT = kernel._utils.DEFAULT_TILES
VMEM = 64 * 2**20                                                      # the kernel's limit (kernel/_utils.py compiler_params). A triple that needs more (rough count, see fits) is not tried
REPS, TRIES = 20, 3                                                    # ms of a triple = (2 x REPS kernels - REPS kernels) / REPS, each time the best of TRIES (see time_tiles)
MAX_SECONDS, INT8_SHARE = float(os.environ.get("MAX_SECONDS", 200)), 0.2   # per model. bf16 states get 1 - INT8_SHARE of it, then int8 states : only the default and the best bf16 triple
OUT = os.path.dirname(tiler_path("x"))                                                       # results/tiler
STATEFUL = bool(kernel.OPTIMIZERS[OPTIM].states)                       # SGD has no state : one pass only (no int8 pass, no state_mode)
MAIN = "bf16 states" if STATEFUL else "no state"                       # what the main pass (key "bf16" in the json) is


def cut(tiles, BT, V, H): return (min(tiles[0], V), min(tiles[1], H), min(tiles[2], BT))

def fits(tiles):
    # rough VMEM of one grid step : acc fp32 + w in, out + m, v in, out (bf16) = 16 B per weight of the tile, + the go and x blocks, twice (the next block is fetched while this one is used)
    bv, bh, bbt = tiles
    return 16 * bv * bh + 2 * 2 * bbt * (bv + bh) <= VMEM

def candidates(BT, V, H):
    # every different triple for this layer, the default first, then by distance from it (in powers of 2)
    seen = {cut(t, BT, V, H) for t in itertools.product(BV, BH, BBT)} | {cut(DEFAULT, BT, V, H)}
    far = lambda t: sum(abs(math.log2(a / b)) for a, b in zip(t, cut(DEFAULT, BT, V, H)))
    return sorted((t for t in seen if fits(t)), key = lambda t: (far(t), t))


#------------------------------------------------------------------------- measure -------------------------------------------------------------------------#

def sync():
    if DEVICE == "tpu": torch.accelerator.synchronize()

def repeat(scalars: jax.Array, count: jax.Array, inputs: list[jax.Array], w: jax.Array, states: list[jax.Array], *, optimizer: str, state_mode: str, tiles: str, interpret: bool) -> tuple[jax.Array, ...]:
    # pure JAX : the fused update of _bridge._run, count[0] times in a row inside one program (each w and states are the input of the next). Returns (w, *states)
    def body(_, carry):
        w, *flat = carry
        out = _bridge._run(scalars, inputs, w, flat, optimizer = optimizer, state_mode = state_mode, interpret = interpret, tiles = tiles)
        return out if len(out) == 1 + len(flat) else (w, *out)                                       # Muon : the kernel gives the states only
    return jax.lax.fori_loop(0, count[0], body, (w, *states))

@functools.cache
def repeat_op(mode, tiles):
    # repeat as a torch op (as _bridge._run_jax_op does for _run), one per (state mode, tiles). w and states are given away : donate_argnums = (3, 4)
    return torch.tpu.pallas.jax_op(f"pallas::forge_tiler_{OPTIM}_{mode}_{tiles}", functools.partial(repeat, optimizer = OPTIM, state_mode = mode, tiles = tiles, interpret = False), donate_argnums = (3, 4))

def update(scalars, count, go, x, w, flat, mode, tiles):
    # count fused updates with these tiles in ONE launch. Returns [w, *states], the old ones are given away
    tiles = "x".join(map(str, tiles))                                  # _bridge takes the tiles as a string
    if DEVICE == "tpu": return list(repeat_op(mode, tiles)(_bridge._device_scalars(tuple(scalars), w.device), torch.tensor([count], dtype = torch.int32, device = w.device), [go, x], w, flat))
    out = repeat(jnp.asarray(scalars, jnp.float32), jnp.asarray([count], jnp.int32), [_bridge._to_jax(go), _bridge._to_jax(x)], _bridge._to_jax(w), [_bridge._to_jax(t) for t in flat],
                 optimizer = OPTIM, state_mode = mode, tiles = tiles, interpret = True)
    return [_bridge._to_torch(a) for a in out]                         # cpu : interpret mode, only to test this file without a TPU

def read(w):
    # waits until every update before is done : one value of the last w is read (each w is the input of the next)
    sync(); return w[0, 0].item()

def time_tiles(layer, mode, tiles):
    # ms of one fused update (kernel only, see the top of the file), or the error text if these tiles do not compile / run
    # layer = (w, go, x) on the CPU. They go to the TPU again for every triple : the update gives w away (donation), and on the TPU w0.clone() can share w0's buffer,
    # so with one w0 on the TPU the first triple gave w0 away and every later triple failed ("a required argument is in a failed state")
    opt = kernel.OPTIMIZERS[OPTIM]
    scalars = opt.scalars(opt.defaults, 10)                            # the same scalars for every kernel : the time does not depend on their values
    try:
        w, go, x = (t.to(DEVICE, DTYPE) for t in layer)
        flat = _bridge._flatten(_bridge.allocate_state(w, OPTIM, mode), OPTIM)
        w, *flat = update(scalars, 1, go, x, w, flat, mode, tiles); read(w)   # the compile (one program for every count : count is data, not a constant)
        best = {}
        for _ in range(TRIES):
            for n in (REPS, 2 * REPS):
                read(w); t0 = time.perf_counter()
                w, *flat = update(scalars, n, go, x, w, flat, mode, tiles); read(w)
                best[n] = min(best.get(n, math.inf), time.perf_counter() - t0)
        return max(best[2 * REPS] - best[REPS], 0.0) * 1e3 / REPS
    except Exception as error: return f"{type(error).__name__}: {str(error)[:200]}"

def measure(name):
    BT, shapes = MODELS[name]
    R, start, path = {"name": name, "optimizer": OPTIM, "BT": BT, "default": list(DEFAULT), "shapes": []}, time.perf_counter(), tiler_path(f"{name}.json")
    left = lambda: MAX_SECONDS - (time.perf_counter() - start)
    for n, ((V, H), count) in enumerate(shapes.items()):
        g = torch.Generator().manual_seed(0)
        layer = (torch.randn(V, H, generator = g) * 0.02, torch.randn(BT, V, generator = g) * 1e-3, torch.randn(BT, H, generator = g))   # w, go, x on the CPU. as a real layer : std 0.02 weights, small go
        errors = 0                                                     # errors in a row
        S = {"V": V, "H": H, "count": count, "bf16": {}, "int8": {}}; R["shapes"].append(S)
        stop = time.perf_counter() + (1 - (INT8_SHARE if STATEFUL else 0)) * left() / (len(shapes) - n)   # this shape's share of the time that is left, less the part kept for int8
        for i, tiles in enumerate(candidates(BT, V, H)):
            if i and time.perf_counter() > stop: break                 # the default triple is always timed
            S["bf16"]["x".join(map(str, tiles))] = ms = time_tiles(layer, "bf16", tiles)
            print(f"{name} {OPTIM} V={V} H={H} {MAIN} tiles {tiles} : " + (f"{ms:8.3f} ms" if isinstance(ms, float) else ms), flush = True)
            json.dump(R, open(path, "w"))                              # after every triple : a crash loses nothing
            errors = 0 if isinstance(ms, float) else errors + 1
            if errors == 3: sys.exit(f"STOP {name} : 3 errors in a row, the TPU state may be broken. The times above are kept")   # do not spend the TPU time on more errors
        ok = {t: ms for t, ms in S["bf16"].items() if isinstance(ms, float)}
        for t in dict.fromkeys([] if not STATEFUL else ["x".join(map(str, cut(DEFAULT, BT, V, H)))] + ([min(ok, key = ok.get)] if ok else [])):   # int8 : the default and the best bf16 triple
            S["int8"][t] = ms = time_tiles(layer, "int8", tuple(map(int, t.split("x"))))
            print(f"{name} {OPTIM} V={V} H={H} int8 tiles {t} : " + (f"{ms:8.3f} ms" if isinstance(ms, float) else ms), flush = True)
            json.dump(R, open(path, "w"))


#------------------------------------------------------------------------- report --------------------------------------------------------------------------#

def table(R):
    # per layer shape : the default against the best triple. Then one triple for the whole model (what DEFAULT_TILES can hold) : sum of count x ms over the layer shapes
    BT, d = R["BT"], tuple(R["default"])
    key = lambda t, S: "x".join(map(str, cut(t, BT, S["V"], S["H"])))
    rows = [f"### {R['name']} ({OPTIM}, BT = {BT}), ms of one fused update, default tiles {d}", "",
            "| layer (V, H) x count | tried | default (ms) | best tiles | best (ms) | gain | int8 default (ms) | int8 with best tiles (ms) |", "|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|"]
    for S in R["shapes"]:
        ok = {t: ms for t, ms in S["bf16"].items() if isinstance(ms, float)}
        base, fmt = ok.get(key(d, S)), lambda ms: f"{ms:.3f}" if isinstance(ms, float) else "failed" if ms else "-"
        best = min(ok, key = ok.get) if ok else None
        rows.append(f"| ({S['V']}, {S['H']}) x {S['count']} | {len(ok)} of {len(S['bf16'])} | {fmt(base)} | {best or '-'} | {fmt(ok.get(best))} | " + (f"{base / ok[best]:.2f}x" if base and best else "-") +
                    f" | {fmt(S['int8'].get(key(d, S)))} | {fmt(S['int8'].get(best))} |")
    total = lambda t: sum(S["count"] * S["bf16"][key(t, S)] for S in R["shapes"]) if all(isinstance(S["bf16"].get(key(t, S)), float) for S in R["shapes"]) else None
    whole = sorted((ms, t) for t in itertools.product(BV, BH, BBT) for ms in [total(t)] if ms is not None)
    rows += ["", f"One triple for every layer of {R['name']} (fused updates of one training step, {MAIN}) : default {d} = " + (f"{total(d):.1f} ms" if total(d) else "not measured on every layer")]
    rows += [f"- {t} : {ms:.1f} ms" + (f" ({total(d) / ms:.2f}x)" if total(d) else "") for ms, t in whole[:3]] or ["- no other triple was measured on every layer (time limit)"]
    return "\n".join(rows), {t: ms / total(d) for ms, t in whole} if total(d) else {}

def report():
    R = [json.load(open(tiler_path(f"{n}.json"))) for n in MODELS if os.path.exists(tiler_path(f"{n}.json"))]
    tables = [table(r) for r in R]
    both = set.intersection(*(set(rel) for _, rel in tables)) if tables else set()   # triples measured on every layer of every model
    rows = [md for md, _ in tables]
    if both:
        best = min(both, key = lambda t: sum(rel[t] for _, rel in tables))
        rows.append(f"### all models\n\nBest single DEFAULT_TILES for {', '.join(r['name'] for r in R)} : {best} (time against the default : " + ", ".join(f"{r['name']} {rel[best]:.2f}" for r, (_, rel) in zip(R, tables)) + ", below 1 is faster)")
    else: rows.append("### all models\n\nNo triple was measured on every layer of every model : use the per-model lines above")
    best = {}
    for r in R:
        for S in r["shapes"]:
            ok = {t: ms for t, ms in S["bf16"].items() if isinstance(ms, float)}
            if ok: best[(r["BT"], S["V"], S["H"])] = tuple(map(int, min(ok, key = ok.get).split("x")))
    rows.append("### per layer shape\n\nThe fastest tiles of every layer shape, for `TILES` in src/forge_tpu/kernel/_utils.py :\n\n```python\nTILES = " + repr(best) + "\n```")
    md = "\n\n".join(rows); open(tiler_path("tiles.md"), "w").write(md + "\n"); print(md)


if __name__ == "__main__":
    # python tiler.py <mlp | llama | qwen> : time the triples of one model (one process each)   |   python tiler.py report : the tables
    os.makedirs(OUT, exist_ok = True)
    if sys.argv[1] == "report": report()
    else: measure(sys.argv[1])
