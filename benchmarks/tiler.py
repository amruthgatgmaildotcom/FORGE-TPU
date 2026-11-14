import itertools, json, math, os, sys, time, torch
import jax.numpy as jnp
from forge_tpu import _bridge, kernel

# Which tiles (bv, bh, bbt) are the fastest ? The kernel uses one fixed triple for every layer (kernel/_utils.py DEFAULT_TILES). This file times many triples on
# every fused layer shape of the three benchmark models : only the fused update of ONE layer (random go, x, w), not a training step, so nothing but the kernel is in the time.
# Each triple is a new XLA compile (the slow part), so there is a time limit : per model MAX_SECONDS, shared equally by its layer shapes. The default triple goes first,
# then the triples closest to it. What was not reached is not in the report. Nothing here changes DEFAULT_TILES : read the report, then change it by hand.
DEVICE, DTYPE = "tpu", torch.bfloat16
MODELS = {"mlp":   (4096, {(16384, 4096): 8, (4096, 16384): 8, (4096, 4096): 1}),                                    # name : (BT, {(V, H) : how many such fused layers})
          "llama": (1024, {(2048, 2048): 32, (512, 2048): 32, (8192, 2048): 32, (2048, 8192): 16}),                 # q o | k v | gate up | down, x 16 layers
          "qwen":  (1024, {(2048, 1024): 28, (1024, 1024): 56, (1024, 2048): 28, (3072, 1024): 56, (1024, 3072): 28})}   # q | k v | o | gate up | down, x 28 layers
BV, BH, BBT = (512, 1024, 2048), (512, 1024, 2048), (512, 1024, 2048)  # every combination is tried (cut to the layer : bv <= V, bh <= H, bbt <= BT)
DEFAULT = kernel._utils.DEFAULT_TILES
VMEM = 64 * 2**20                                                      # the kernel's limit (kernel/_utils.py compiler_params). A triple that needs more (rough count, see fits) is not tried
REPS, WARM = 5, 2                                                      # ms of a triple = REPS updates in a row / REPS, after WARM updates (the first one compiles)
MAX_SECONDS, INT8_SHARE = float(os.environ.get("MAX_SECONDS", 200)), 0.2   # per model. bf16 states get 1 - INT8_SHARE of it, then int8 states : only the default and the best bf16 triple
OUT = os.environ.get("RESULTS", "results")


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

def update(scalars, go, x, w, flat, mode, tiles):
    tiles = "x".join(map(str, tiles))                                  # _bridge takes the tiles as a string
    # one fused update with these tiles, as _bridge.fused_update does it (without the replay, the same for every triple). Returns [w, *states], the old ones are given away
    if DEVICE == "tpu": return _bridge._run_jax_op("adamw", mode, tiles)(_bridge._device_scalars(tuple(scalars), w.device), [go, x], w, flat)
    out = _bridge._run(jnp.asarray(scalars, jnp.float32), [_bridge._to_jax(go), _bridge._to_jax(x)], _bridge._to_jax(w), [_bridge._to_jax(t) for t in flat], optimizer = "adamw", state_mode = mode, interpret = True, tiles = tiles)
    return [_bridge._to_torch(a) for a in out]                         # cpu : interpret mode, only to test this file without a TPU

def read(w):
    # torch_tpu only queues a fused update : it runs when something needs its result. Reading one value of the last w runs every update before it (each w is the input of the next)
    sync(); return w[0, 0].item()

def time_tiles(go, x, w0, mode, tiles):
    # ms of one fused update (REPS updates in a row, then one read, / REPS), or the error text if these tiles do not compile / run
    opt = kernel.OPTIMIZERS["adamw"]
    try:
        w = w0.clone()
        flat = _bridge._flatten(_bridge.allocate_state(w, "adamw", mode), "adamw")
        for step in range(1, WARM + 1): w, *flat = update(opt.scalars(opt.defaults, step), go, x, w, flat, mode, tiles)
        read(w); t0 = time.perf_counter()                              # the compile and the warm updates are done here
        for step in range(WARM + 1, WARM + REPS + 1): w, *flat = update(opt.scalars(opt.defaults, step), go, x, w, flat, mode, tiles)
        read(w); return (time.perf_counter() - t0) * 1e3 / REPS
    except Exception as error: return f"{type(error).__name__}: {str(error)[:200]}"

def measure(name):
    BT, shapes = MODELS[name]
    R, start, path = {"name": name, "BT": BT, "default": list(DEFAULT), "shapes": []}, time.perf_counter(), f"{OUT}/tiler_{name}.json"
    left = lambda: MAX_SECONDS - (time.perf_counter() - start)
    for n, ((V, H), count) in enumerate(shapes.items()):
        g = torch.Generator().manual_seed(0)
        w0 = (torch.randn(V, H, generator = g) * 0.02).to(DEVICE, DTYPE)                                             # as a real layer : std 0.02 weights, small go
        go, x = (torch.randn(BT, V, generator = g) * 1e-3).to(DEVICE, DTYPE), torch.randn(BT, H, generator = g).to(DEVICE, DTYPE)
        S = {"V": V, "H": H, "count": count, "bf16": {}, "int8": {}}; R["shapes"].append(S)
        stop = time.perf_counter() + (1 - INT8_SHARE) * left() / (len(shapes) - n)   # this shape's share of the time that is left, less the part kept for int8
        for i, tiles in enumerate(candidates(BT, V, H)):
            if i and time.perf_counter() > stop: break                 # the default triple is always timed
            S["bf16"]["x".join(map(str, tiles))] = ms = time_tiles(go, x, w0, "bf16", tiles)
            print(f"{name} V={V} H={H} bf16 tiles {tiles} : " + (f"{ms:8.3f} ms" if isinstance(ms, float) else ms), flush = True)
            json.dump(R, open(path, "w"))                              # after every triple : a crash loses nothing
        ok = {t: ms for t, ms in S["bf16"].items() if isinstance(ms, float)}
        for t in dict.fromkeys(["x".join(map(str, cut(DEFAULT, BT, V, H)))] + ([min(ok, key = ok.get)] if ok else [])):   # int8 : the default and the best bf16 triple
            S["int8"][t] = ms = time_tiles(go, x, w0, "int8", tuple(map(int, t.split("x"))))
            print(f"{name} V={V} H={H} int8 tiles {t} : " + (f"{ms:8.3f} ms" if isinstance(ms, float) else ms), flush = True)
            json.dump(R, open(path, "w"))
        del w0, go, x


#------------------------------------------------------------------------- report --------------------------------------------------------------------------#

def table(R):
    # per layer shape : the default against the best triple. Then one triple for the whole model (what DEFAULT_TILES can hold) : sum of count x ms over the layer shapes
    BT, d = R["BT"], tuple(R["default"])
    key = lambda t, S: "x".join(map(str, cut(t, BT, S["V"], S["H"])))
    rows = [f"### {R['name']} (BT = {BT}), ms of one fused update, default tiles {d}", "",
            "| layer (V, H) x count | tried | default (ms) | best tiles | best (ms) | gain | int8 default (ms) | int8 with best tiles (ms) |", "|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|"]
    for S in R["shapes"]:
        ok = {t: ms for t, ms in S["bf16"].items() if isinstance(ms, float)}
        base, fmt = ok.get(key(d, S)), lambda ms: f"{ms:.3f}" if isinstance(ms, float) else "failed" if ms else "-"
        best = min(ok, key = ok.get) if ok else None
        rows.append(f"| ({S['V']}, {S['H']}) x {S['count']} | {len(ok)} of {len(S['bf16'])} | {fmt(base)} | {best or '-'} | {fmt(ok.get(best))} | " + (f"{base / ok[best]:.2f}x" if base and best else "-") +
                    f" | {fmt(S['int8'].get(key(d, S)))} | {fmt(S['int8'].get(best))} |")
    total = lambda t: sum(S["count"] * S["bf16"][key(t, S)] for S in R["shapes"]) if all(isinstance(S["bf16"].get(key(t, S)), float) for S in R["shapes"]) else None
    whole = sorted((ms, t) for t in itertools.product(BV, BH, BBT) for ms in [total(t)] if ms is not None)
    rows += ["", f"One triple for every layer of {R['name']} (fused updates of one training step, bf16 states) : default {d} = " + (f"{total(d):.1f} ms" if total(d) else "not measured on every layer")]
    rows += [f"- {t} : {ms:.1f} ms" + (f" ({total(d) / ms:.2f}x)" if total(d) else "") for ms, t in whole[:3]] or ["- no other triple was measured on every layer (time limit)"]
    return "\n".join(rows), {t: ms / total(d) for ms, t in whole} if total(d) else {}

def report():
    R = [json.load(open(f"{OUT}/tiler_{n}.json")) for n in MODELS if os.path.exists(f"{OUT}/tiler_{n}.json")]
    tables = [table(r) for r in R]
    both = set.intersection(*(set(rel) for _, rel in tables)) if tables else set()   # triples measured on every layer of every model
    rows = [md for md, _ in tables]
    if both:
        best = min(both, key = lambda t: sum(rel[t] for _, rel in tables))
        rows.append(f"### all models\n\nBest single DEFAULT_TILES for {', '.join(r['name'] for r in R)} : {best} (time against the default : " + ", ".join(f"{r['name']} {rel[best]:.2f}" for r, (_, rel) in zip(R, tables)) + ", below 1 is faster)")
    else: rows.append("### all models\n\nNo triple was measured on every layer of every model : use the per-model lines above")
    md = "\n\n".join(rows); open(f"{OUT}/tiler_table.md", "w").write(md + "\n"); print(md)


if __name__ == "__main__":
    # python tiler.py <mlp | llama | qwen> : time the triples of one model (one process each)   |   python tiler.py report : the tables
    os.makedirs(OUT, exist_ok = True)
    if sys.argv[1] == "report": report()
    else: measure(sys.argv[1])
