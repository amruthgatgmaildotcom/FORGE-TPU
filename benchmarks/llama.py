import torch, torch.nn as nn, torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM
from forge_tpu import FusedLinear
from mlp import OPTIM, OPTIMIZERS, cli, fuse, torch_optimizer                                        # the shared measure + report engine (same metrics, plots and colours for every model)

# Llama-3.2-1B : the model code is LlamaForCausalLM of the transformers library, built from the numbers of the official config.json (1.24B params, tied embedding).
# The weights are NOT downloaded (2.5 GB, and the meta-llama repo needs an access token) : random init by transformers (std 0.02, norms 1). Step time and memory do not depend on the values.
# Data : a random Markov chain on ACTIVE tokens of the full vocab, each token has K possible next tokens, so the loss falls from ln(V) = 11.8 toward ln(K) = 1.4,
# and then below it : the BATCHES sequences are fixed, the model learns them by heart
DEVICE, DTYPE = "tpu", torch.bfloat16
V, D, FF, LAYERS, HEADS, KV_HEADS, THETA, EPS = 128256, 2048, 8192, 16, 32, 8, 500000.0, 1e-5
BATCH, SEQ, BATCHES, ACTIVE, K = 1, 1024, 8, 512, 4                    # BATCH x SEQ tokens per step
WD = 0.01
LR = {"adamw": 1e-4, "sgd": 1e-2}[OPTIM]                         # OPTIM : the optimizer of this run (benchmarks/methods.py)
HD = D // HEADS


def Llama():
    # config.json of meta-llama/Llama-3.2-1B, without its rope_scaling (it only changes the RoPE angles, not the work) and without the generation settings.
    # use_cache = False : training, no key / value cache is kept
    return LlamaForCausalLM(LlamaConfig(vocab_size = V, hidden_size = D, intermediate_size = FF, num_hidden_layers = LAYERS, num_attention_heads = HEADS, num_key_value_heads = KV_HEADS, head_dim = HD,
                                        hidden_act = "silu", rope_theta = THETA, rms_norm_eps = EPS, max_position_embeddings = 131072, initializer_range = 0.02, tie_word_embeddings = True, use_cache = False))


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
    model, params = Llama().to(DEVICE, DTYPE), None
    model.lm_head.weight = model.model.embed_tokens.weight             # tie again : .to() a TPU makes a new Parameter for each module, so the tie breaks
    if name.startswith("forge"):
        fuse(model.model.layers, name, LR, WD)
        fused = {id(m.weight) for m in model.modules() if isinstance(m, FusedLinear)}
        params = [p for p in model.parameters() if id(p) not in fused]
    return model, torch_optimizer(name, params or model.parameters(), LR, WD)


def loss_fn(out, y): return F.cross_entropy(out.logits.flatten(0, 1).float(), y.flatten())   # out : transformers' output object


if __name__ == "__main__":
    cli("llama", make_model, make_data, loss_fn, lambda params: (6 * params + 12 * LAYERS * SEQ * D) * BATCH * SEQ,   # 6 N per token (tied matrix once) + attention
        f"Llama-3.2-1B (transformers), {{params}}B params (random init), batch {BATCH}, sequence {SEQ}", "Loss (cross-entropy)")
