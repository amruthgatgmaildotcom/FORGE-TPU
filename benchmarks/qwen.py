import torch, torch.nn as nn, torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM
from forge_tpu import FusedLinear
from mlp import OPTIM, OPTIMIZERS, cli, fuse, torch_optimizer                                        # the shared measure + report engine (same metrics, plots and colours for every model)

# Qwen3-0.6B : the model code is Qwen3ForCausalLM of the transformers library (RMSNorm on each query / key head), built from the numbers of the official config.json
# (0.60B params, tied embedding). The weights are NOT downloaded (1.5 GB) : random init by transformers (std 0.02, norms 1). Step time and memory do not depend on the values.
# Qwen3-0.6B is the largest Qwen3 that fits vanilla AdamW in 16 GB (Qwen3-1.7B : 2.0B params x 8 B = 16 GB before activations).
# Data : a random Markov chain on ACTIVE tokens of the full vocab, each token has K possible next tokens, so the loss falls from ln(V) = 11.9 toward ln(K) = 1.4
# (and below it after enough steps : the BATCHES sequences are fixed, the model learns them by heart)
DEVICE, DTYPE = "tpu", torch.bfloat16
V, D, FF, LAYERS, HEADS, KV_HEADS, HD, THETA, EPS = 151936, 1024, 3072, 28, 16, 8, 128, 1000000.0, 1e-6
BATCH, SEQ, BATCHES, ACTIVE, K = 1, 1024, 8, 512, 4                    # BATCH x SEQ tokens per step
WD = 0.01
LR = {"adamw": 1e-4, "sgd": 1e-2}[OPTIM]                         # OPTIM : the optimizer of this run (benchmarks/methods.py)


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


def make_model(name, seed = 0):
    # FORGE fuses every Linear inside the layers. The tied embedding / head and the norms stay on the fused torch optimizer of the same family (AdamW or SGD)
    torch.manual_seed(seed)                                            # same init for all four optimizers
    model, params = Qwen().to(DEVICE, DTYPE), None
    model.lm_head.weight = model.model.embed_tokens.weight             # tie again : .to() a TPU makes a new Parameter for each module, so the tie breaks
    if name.startswith("forge"):
        fuse(model.model.layers, name, LR, WD)
        fused = {id(m.weight) for m in model.modules() if isinstance(m, FusedLinear)}
        params = [p for p in model.parameters() if id(p) not in fused]
    return model, torch_optimizer(name, params or model.parameters(), LR, WD)


def loss_fn(out, y): return F.cross_entropy(out.logits.flatten(0, 1).float(), y.flatten())   # out : transformers' output object


if __name__ == "__main__":
    cli("qwen", make_model, make_data, loss_fn, lambda params: (6 * params + 12 * LAYERS * SEQ * HEADS * HD) * BATCH * SEQ,   # 6 N per token (tied matrix once) + attention
        f"Qwen3-0.6B (transformers), {{params}}B params (random init), batch {BATCH}, sequence {SEQ}", "Loss (cross-entropy)")
