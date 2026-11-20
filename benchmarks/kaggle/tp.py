import os, sys, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist

#   Readme.md
#     Tensor parallel (TP) pieces for the 8-chip runs of Llama-3.1-8B (TPU v5e-8 : 8 chips, 16 GiB HBM each, one process per chip). src/ is NOT changed :
#     every chip holds normal nn.Linear / FusedLinear layers, each one with 1/8 of the full weight. FORGE runs on each chip on its own shard.
#     Megatron layout, per block :
#        norm -> Copy -> column-parallel Linears (q, k, v / gate, up / lm_head : out_features / 8)    Copy   : forward identity, backward all-reduce (sum) of dx
#             -> row-parallel Linear (o / down : in_features / 8) -> Reduce                           Reduce : forward all-reduce (sum), backward identity
#     dW of a column shard = go[:, shard].T @ x (x is the same on every chip), dW of a row shard = go.T @ x[:, shard] (go is the same on every chip) : FORGE needs nothing else.
#     The embedding and the logits are split over the vocabulary (VocabEmbedding, vocab_cross_entropy). Norm weights are the same on every chip (their gradient is too).
#     Every chip runs the same steps on the same tokens, so all the collectives match. Weights : each shard has its own seeded generator, so a run is the same for
#     every method, and the full matrix can be built on one chip (the TP check in run.py smoke).

DEVICE = os.environ.get("TP_DEVICE", "tpu")                           # "cpu" : only for the CPU simulation of this code (gloo, 8 processes)
DTYPE = torch.bfloat16
RANK, WORLD = 0, 1



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def init():
    # one process per chip, started by torchrun (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT, LOCAL_RANK in the environment)
    global RANK, WORLD
    import datetime                                                    # no practical time limit on a collective (torch's default : 10 to 30 min)
    dist.init_process_group(backend = "gloo" if DEVICE == "cpu" else "tpu_dist", timeout = datetime.timedelta(days = 1))
    RANK, WORLD = dist.get_rank(), dist.get_world_size()
    return RANK, WORLD

def all_sum(t):
    t = t.contiguous().clone(); dist.all_reduce(t); return t           # SUM is the default op

def all_gather(t):
    # (n,) on each chip -> (WORLD, n), row r from chip r
    t = t.contiguous(); out = torch.empty(WORLD * t.numel(), dtype = t.dtype, device = t.device)
    dist.all_gather_into_tensor(out, t.flatten()); return out.view(WORLD, *t.shape)

def barrier():
    all_sum(torch.ones(1, device = DEVICE)); torch.accelerator.synchronize() if DEVICE != "cpu" else None

class Copy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return all_sum(g)

class Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return all_sum(x)
    @staticmethod
    def backward(ctx, g): return g

class Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return all_gather(x)
    @staticmethod
    def backward(ctx, g): return g[RANK].contiguous()                  # every chip computes the same thing after the gather : its own row is its gradient



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



class NormCopy(nn.Module):
    # norm (replicated weight), then Copy : the input of column-parallel Linears
    def __init__(self, norm, tp): super().__init__(); self.norm, self.tp = norm, tp
    def forward(self, x):
        y = self.norm(x)
        return Copy.apply(y) if self.tp else y

class RowOut(nn.Module):
    # a row-parallel Linear (its input is split), then Reduce : the sum of the 8 partial outputs
    def __init__(self, linear, tp): super().__init__(); self.linear, self.tp = linear, tp
    def forward(self, x):
        y = self.linear(x)
        return Reduce.apply(y) if self.tp else y

class VocabEmbedding(nn.Module):
    # rows [start, start + n) of the embedding on this chip : the other ids give 0 here, the sum over the chips gives every row
    def __init__(self, weight, start, tp): super().__init__(); self.weight, self.start, self.tp = nn.Parameter(weight), start, tp
    def forward(self, ids):
        local = ids - self.start
        outside = (local < 0) | (local >= self.weight.shape[0])
        out = F.embedding(local.masked_fill(outside, 0), self.weight).masked_fill(outside.unsqueeze(-1), 0)
        return Reduce.apply(out) if self.tp else out

def vocab_cross_entropy(logits, y, start, tp):
    # = F.cross_entropy(full logits, y) (mean), with logits (B, T, V / 8) of this chip : logsumexp of each chip, gathered, and the target logit from its chip
    z = logits.flatten(0, 1).float()
    lse = torch.logsumexp(z, dim = -1)
    if tp: lse = torch.logsumexp(Gather.apply(lse), dim = 0)
    t = y.flatten() - start
    outside = (t < 0) | (t >= z.shape[1])
    target = z.gather(1, t.clamp(0, z.shape[1] - 1).unsqueeze(1)).squeeze(1).masked_fill(outside, 0)
    if tp: target = Reduce.apply(target)
    return (lse - target).mean()



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



class Init:
    # The weights of one model. Each matrix has an index (the order it is made in) and is split in WORLD shards along dim ("out" : rows, "in" : columns).
    # Shard q of matrix i comes from its own CPU generator (seed, i, q), so it is the same in every run and on every chip that makes it.
    #   tp = True  : this chip's shard (RANK)       tp = False : the full matrix (the WORLD shards, joined) : the one-chip reference of the TP check
    def __init__(self, seed, tp, world = None): self.seed, self.tp, self.world, self.index = seed, tp, world or WORLD, 0
    def matrix(self, rows, cols, dim, kind, scale):
        # kind "normal" : N(0, scale^2) (transformers' init, std 0.02)  |  "uniform" : U(-scale, scale) (nn.Linear's init, scale = 1 / sqrt(in_features))
        i, self.index = self.index, self.index + 1
        if dim == "out": assert rows % self.world == 0, (rows, self.world); shape = (rows // self.world, cols)
        else: assert cols % self.world == 0, (cols, self.world); shape = (rows, cols // self.world)
        def shard(q):
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + i * 1_009 + q)
            t = torch.randn(shape, generator = g) * scale if kind == "normal" else (torch.rand(shape, generator = g) * 2 - 1) * scale
            return t.to(DEVICE, DTYPE)
        if self.tp: return shard(RANK)
        return torch.cat([shard(q) for q in range(self.world)], dim = 0 if dim == "out" else 1)
    def linear(self, in_features, out_features, dim, kind = "normal", scale = 0.02):
        w = self.matrix(out_features, in_features, dim, kind, scale)
        with torch.device("meta"): lin = nn.Linear(w.shape[1], w.shape[0], bias = False)
        lin.weight = nn.Parameter(w)
        lin.weight.tp_dim = (0 if dim == "out" else 1) if self.tp else None   # the split dimension of this shard (Muon gathers the full matrix with it), None : the full matrix
        return lin
    def start(self, size):
        # the first vocabulary row of this chip
        return RANK * (size // self.world) if self.tp else 0



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def build_lm(model_class, rotary_class, config, tp, seed = 0):
    # LlamaForCausalLM / Qwen3ForCausalLM of transformers, made on the meta device (no memory), then every Linear, the embedding and the norms are made real :
    # with tp, this chip's shards (Megatron layout above), without tp, the full model (the reference of the TP check). The model code of transformers is not changed :
    # its attention reshapes with -1 heads, so it runs on the 4 query heads / 1 key-value head of this chip as it runs on 32 / 8
    c = config
    D, H, KV, HD, FF, V = c.hidden_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.intermediate_size, c.vocab_size
    init, shared_by_heads = Init(seed, tp), []
    with torch.device("meta"): model = model_class(c)
    for layer in model.model.layers:
        a, m = layer.self_attn, layer.mlp
        a.q_proj, a.k_proj, a.v_proj = init.linear(D, H * HD, "out"), init.linear(D, KV * HD, "out"), init.linear(D, KV * HD, "out")
        a.o_proj = RowOut(init.linear(H * HD, D, "in"), tp)
        m.gate_proj, m.up_proj = init.linear(D, FF, "out"), init.linear(D, FF, "out")
        m.down_proj = RowOut(init.linear(FF, D, "in"), tp)
        layer.input_layernorm, layer.post_attention_layernorm = NormCopy(layer.input_layernorm, tp), NormCopy(layer.post_attention_layernorm, tp)
        shared_by_heads += [getattr(a, n) for n in ("q_norm", "k_norm") if hasattr(a, n)]   # Qwen3 : one RMSNorm weight for every head, the heads are split
    model.model.norm = NormCopy(model.model.norm, tp)
    model.model.embed_tokens = VocabEmbedding(init.matrix(V, D, "out", "normal", 0.02), init.start(V), tp)
    model.lm_head = init.linear(D, V, "out")
    model.vocab_start = init.start(V)
    model.model.rotary_emb = rotary_class(config = c).to(DEVICE, DTYPE)   # as the 1-chip benchmarks : model.to(DEVICE, DTYPE) casts its inv_freq to bf16 too
    materialize_norms(model)
    if tp:
        # each chip sees only its own heads, so the gradient of a weight shared by all heads is a part : the sum over the chips is the full gradient
        for norm in shared_by_heads: norm.weight.register_hook(all_sum)
    return model

def materialize_norms(model):
    # what is still on the meta device must be a norm weight : ones (as transformers' init). Anything else is a bug here
    for module_name, module in model.named_modules():
        for name, p in list(module._parameters.items()):
            if p is not None and p.is_meta:
                if "norm" not in f"{module_name}.{name}": raise RuntimeError(f"tp.py : {module_name}.{name} was not made")
                module._parameters[name] = nn.Parameter(torch.ones(p.shape, dtype = DTYPE, device = DEVICE))
        for name, b in module._buffers.items():
            if b is not None and b.is_meta: raise RuntimeError(f"tp.py : buffer {module_name}.{name} was not made")

def lm_params(c):
    # parameters of the full model (not of one chip)
    D, H, KV, HD, FF, V, L = c.hidden_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.intermediate_size, c.vocab_size, c.num_hidden_layers
    qk_norms = 2 * HD if "qwen3" in c.model_type else 0
    return 2 * V * D + D + L * (D * H * HD * 2 + 2 * D * KV * HD + 3 * D * FF + 2 * D + qk_norms)



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def full_matrix(w, B):
    # the full matrix of B, a shard split like the weight w (tp_dim), on every chip (one all_gather)
    d = getattr(w, "tp_dim", None)
    if d is None: return B
    g = all_gather(B)                                                  # (WORLD, *B.shape), row q = chip q's shard
    return g.reshape(WORLD * B.shape[0], B.shape[1]) if d == 0 else torch.cat(list(g), dim = 1)

def my_part(O, w):
    d = getattr(w, "tp_dim", None)
    return O if d is None else O.narrow(d, RANK * w.shape[d], w.shape[d])

@torch.no_grad()
def apply_muon(items):
    # Muon's step after backward, tensor parallel (forge_tpu.muon.apply_muon for shards). Newton-Schulz needs the full matrix (FORGE paper App. I.1 : "gathered
    # Newton-Schulz is bit-identical to single-device"). For each group of WORLD matrices : every chip gathers their full momenta, chip j orthogonalizes matrix j
    # (the work is split, not repeated 8 times), a sum over the chips (the others add zeros) gives every chip the result, each chip updates its own shard.
    # The same function for FORGE Muon and the baseline. items : [(w, B, lr, weight_decay, ns_steps)], in the same order on every chip
    from forge_tpu.muon import newton_schulz, muon_update
    for start in range(0, len(items), WORLD):
        group = items[start:start + WORLD]
        full = [full_matrix(w, B) for w, B, *_ in group]
        mine = newton_schulz(full[RANK], group[RANK][4]) if RANK < len(group) else None
        for j, (w, B, lr, weight_decay, steps) in enumerate(group):
            O = all_sum(mine if j == RANK else torch.zeros(full[j].shape, dtype = torch.bfloat16, device = full[j].device))
            muon_update(w, my_part(O, w), lr, weight_decay, full_shape = tuple(full[j].shape))
        del full, mine

def muon_classes(tp_on):
    # the Muon classes of benchmarks/methods.py's make_optimizer : with tensor parallel, their Newton-Schulz step is apply_muon above
    import functools
    from forge_tpu import Muon, MuonStep
    return dict(muon = functools.partial(Muon, apply = apply_muon), muon_step = functools.partial(MuonStep, apply = apply_muon)) if tp_on else {}

def quiet_other_ranks(log_dir):
    # only chip 0 prints : the output of the other chips goes to <log_dir>/rank<r>.log (the cell prints their end if a run fails)
    if RANK == 0: return
    os.makedirs(log_dir, exist_ok = True)
    f = open(os.path.join(log_dir, f"rank{RANK}.log"), "w", buffering = 1)
    os.dup2(f.fileno(), 1); os.dup2(f.fileno(), 2)
    sys.stdout = sys.stderr = f
