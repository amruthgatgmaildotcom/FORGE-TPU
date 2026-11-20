import torch, torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM
from config import OPTIM, SEQ
from methods import MUON_LR, fuse, make_optimizer

#   Readme.md
#     Qwen3-0.6B : Qwen3ForCausalLM of the transformers library (RMSNorm on each query / key head), built from the numbers of the official config.json (0.60B params,
#     tied embedding). The weights are NOT downloaded : random init by transformers (std 0.02, norms 1). The largest Qwen3 that fits vanilla AdamW in 16 GB.
#     Data : as llama.py, a random Markov chain on ACTIVE tokens of the full vocabulary, BATCHES fixed sequences, cycled.
#     FORGE takes every Linear inside the layers; the tied embedding / head and the norms stay on the fused torch optimizer (fused AdamW for Muon)

DEVICE, DTYPE = "tpu", torch.bfloat16
V, D, FF, LAYERS, HEADS, KV_HEADS, HD, THETA, EPS = 151936, 1024, 3072, 28, 16, 8, 128, 1000000.0, 1e-6
BATCH, ACTIVE, K, BATCHES = 1, 512, 4, 8                                                     # BATCH x SEQ tokens per step (SEQ : config.py)
WD = 0.01
LR = {"adamw": 1e-4, "sgd": 1e-2, "muon": 1e-4}[OPTIM]
SETTINGS = dict(tokens_per_step = BATCH * SEQ, batch = BATCH, seq = SEQ, batches = BATCHES, lr = LR, weight_decay = WD)



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def Qwen():
    # config.json of Qwen/Qwen3-0.6B, without the generation settings. use_cache = False : training, no key / value cache is kept
    return Qwen3ForCausalLM(Qwen3Config(vocab_size = V, hidden_size = D, intermediate_size = FF, num_hidden_layers = LAYERS, num_attention_heads = HEADS, num_key_value_heads = KV_HEADS, head_dim = HD,
                                        hidden_act = "silu", rope_theta = THETA, rms_norm_eps = EPS, max_position_embeddings = 40960, initializer_range = 0.02, tie_word_embeddings = True, use_cache = False))

def make_data(seed = 0):
    # BATCHES fixed (input, target) pairs on DEVICE, the same for every run
    g = torch.Generator().manual_seed(seed)
    vocab, nxt = torch.randperm(V, generator = g)[:ACTIVE], torch.randint(ACTIVE, (ACTIVE, K), generator = g)
    s = torch.empty(BATCHES * BATCH, SEQ + 1, dtype = torch.long)
    s[:, 0] = torch.randint(ACTIVE, (BATCHES * BATCH,), generator = g)
    for t in range(SEQ): s[:, t + 1] = nxt[s[:, t], torch.randint(K, (BATCHES * BATCH,), generator = g)]
    ids = vocab[s].view(BATCHES, BATCH, SEQ + 1).to(DEVICE)
    return [(b[:, :-1].contiguous(), b[:, 1:].contiguous()) for b in ids]

def make_model(method, seed = 0):
    torch.manual_seed(seed)                                                                  # the same init for every method
    model = Qwen().to(DEVICE, DTYPE)
    model.lm_head.weight = model.model.embed_tokens.weight                                   # tie again : .to() a TPU makes a new Parameter for each module, so the tie breaks
    if method.startswith("forge"): fuse(model.model.layers, method, LR, WD)
    return model, make_optimizer(method, model, model.model.layers, LR, WD)

def loss_fn(out, y): return F.cross_entropy(out.logits.flatten(0, 1).float(), y.flatten())   # out : transformers' output object
def flops(params): return (6 * params + 12 * LAYERS * SEQ * HEADS * HD) * BATCH * SEQ                                                            # 6 N per token (the tied matrix once) + attention
