import jax
from jax.experimental.pallas import tpu as pltpu
from .codecs import CODECS
from .optimizers import OPTIMIZERS

DEFAULT_TILES = (512, 2048, 2048)     # (bv, bh, bbt) : the best single triple of benchmarks/tiler.py for AdamW and SGD on the three models
#   Readme.md
#      This file is internal to the kernel folder (starts with _), used by fused_grad.py (and lookup by __init__.py)
#      each defined function is readable and self explainatory, so im skipping explaination here




















#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def lookup(optimizer, state_mode):
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {optimizer!r}; choose from {sorted(OPTIMIZERS)}")
    if state_mode not in CODECS:
        raise ValueError(f"unsupported state_mode {state_mode!r}; choose from {sorted(CODECS)}")
    return OPTIMIZERS[optimizer], CODECS[state_mode]

def _check(name, block, dim, multiple):
    if block != dim and (block <= 0 or block % multiple):
        raise ValueError(f"tile {name}={block} must equal {dim} or be a positive multiple of {multiple}")
    # DOCUMENTATION 
    # TPUs perform the bulk of the computation on 2D vector registers, which are typically of size 8x128 for 32-bit values
    # (as of TPU v6). When a vector value is loaded from VMEM into registers (e.g. x = x_ref[...]), the last two dimensions
    # of the array will be tiled into the registers. Pallas will only ever consider mapping the last two dimensions of
    # intermediate arrays to the 8x128 vector register dimensions (sublanes and lanes respectively). 
    # This holds for v5 also ==> (bv % 128 = 0, bh % 128 = 0, bbt % 8 = 0)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



TILES = {  # {(BT, V, H): (bv, bh, bbt)} : the tiles of one layer shape, DEFAULT_TILES for every other shape (benchmarks/tiler.py prints this dict from measured times)
    # Llama-3.1-8B on 8 chips (kaggle/tp.py), the layer shapes of one chip. NOT measured yet : chosen with an HBM traffic + pipeline model (12 B/param moved, x re-read once per row
    # block, go once per column block, the first load and the last store not hidden, 0.35 us per grid step). The gain : gate / up (V = 1792) are one row block, the default bv = 512
    # gave a half-empty last block (1/8 of the traffic wasted); q and k / v get more, smaller column blocks, so the pipeline hides more of the loads. Same results as the default tiles.
    (512, 512, 4096): (512, 512, 512),    (512, 128, 4096): (128, 1024, 512),    (512, 1792, 4096): (1792, 512, 512),
    (1024, 512, 4096): (512, 1024, 1024), (1024, 128, 4096): (128, 1024, 1024),  (1024, 4096, 512): (1024, 512, 1024), (1024, 1792, 4096): (1792, 512, 1024)}
def tiles_for(BT, V, H, tiles, codec):
  # the tiles of one call : given ones, else the measured best of this shape (TILES), else DEFAULT_TILES. A TILES entry the state format cannot use (int8 : bv % 512) falls back to the default
    if tiles is None and (BT, V, H) in TILES:
        try: return (*pick_tiles(V, H, TILES[(BT, V, H)], codec), pick_bt_tile(BT, TILES[(BT, V, H)]))
        except ValueError: pass
    return (*pick_tiles(V, H, tiles, codec), pick_bt_tile(BT, tiles))
def pick_tiles(V, H, tiles, codec):
    bv, bh = tiles[:2] if tiles else (min(V, DEFAULT_TILES[0]), min(H, DEFAULT_TILES[1]))
    _check("bv", bv, V, 128); _check("bh", bh, H, 128); codec.check(V, bv)
    return bv, bh

def pick_bt_tile(BT, tiles):
    bbt = tiles[2] if tiles else min(BT, DEFAULT_TILES[2])
    _check("bbt", bbt, BT, 8)
    return bbt



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def like(w, states):
    # pallas_call's out_shape: the shape and dtype (no data) of w and of every state array.
    def shape_of(a): return jax.ShapeDtypeStruct(a.shape, a.dtype)
    return shape_of(w), {s: {name: shape_of(a) for name, a in arrays.items()} for s, arrays in states.items()}

def count_arrays(states):
    # This function returns totally how many state matrices are we maintaining 
    return sum(len(arrays) for arrays in states.values())

def compiler_params(semantics):
    # v5e caps a kernel's VMEM at 16 MiB by default, (128 MiB exist), setting it to 64MiB (MAX).
    return pltpu.CompilerParams(dimension_semantics = semantics, vmem_limit_bytes = 64*(2**20))



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

