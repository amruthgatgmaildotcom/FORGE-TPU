import torch, torch.nn as nn, torch.nn.functional as F
from forge_tpu import FusedLinear
from mlp import OPTIMIZERS, cli                                        # the shared measure + report engine (same metrics, plots and colours for every model)

# Llama-3.2-1B architecture (1.24B params, tied embedding), random init as in Llama (std 0.02, norms 1) : nothing to download.
# Data : a random Markov chain on ACTIVE tokens of the full vocab, each token has K possible next tokens, so the loss falls from ln(V) = 11.8 toward ln(K) = 1.4
DEVICE, DTYPE = "tpu", torch.bfloat16
V, D, FF, LAYERS, HEADS, KV_HEADS, THETA, EPS = 128256, 2048, 8192, 16, 32, 8, 500000.0, 1e-5
BATCH, SEQ, BATCHES, ACTIVE, K = 1, 1024, 8, 512, 4                    # BATCH x SEQ tokens per step
LR, WD = 1e-4, 0.01
HD = D // HEADS


def rope(x, cos, sin):
    x1, x2 = x.chunk(2, -1)
    return x * cos + torch.cat((-x2, x1), -1) * sin


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q, self.k, self.v, self.o = nn.Linear(D, D, False), nn.Linear(D, KV_HEADS * HD, False), nn.Linear(D, KV_HEADS * HD, False), nn.Linear(D, D, False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q, k, v = (proj(x).view(B, T, -1, HD).transpose(1, 2) for proj in (self.q, self.k, self.v))
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        k, v = (t.repeat_interleave(HEADS // KV_HEADS, 1) for t in (k, v))   # grouped-query attention : 8 KV heads serve 32 query heads
        return self.o(F.scaled_dot_product_attention(q, k, v, is_causal = True).transpose(1, 2).reshape(B, T, D))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate, self.up, self.down = nn.Linear(D, FF, False), nn.Linear(D, FF, False), nn.Linear(FF, D, False)

    def forward(self, x): return self.down(F.silu(self.gate(x)) * self.up(x))


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn, self.ffn, self.norm1, self.norm2 = Attention(), FeedForward(), nn.RMSNorm(D, EPS), nn.RMSNorm(D, EPS)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.ffn(self.norm2(x))


class Llama(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed, self.layers, self.norm = nn.Embedding(V, D), nn.ModuleList(Layer() for _ in range(LAYERS)), nn.RMSNorm(D, EPS)
        self.head = nn.Linear(D, V, False)
        self.head.weight = self.embed.weight                               # tied, as in Llama-3.2-1B
        angles = torch.outer(torch.arange(SEQ), THETA ** (-torch.arange(0, HD, 2) / HD)).repeat(1, 2)
        self.register_buffer("cos", angles.cos(), persistent = False); self.register_buffer("sin", angles.sin(), persistent = False)
        for p in self.parameters():
            if p.dim() == 2: nn.init.normal_(p, std = 0.02)

    def forward(self, ids):
        x = self.embed(ids)
        for layer in self.layers: x = layer(x, self.cos, self.sin)
        return self.head(self.norm(x))


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
    # FORGE fuses every Linear inside the layers. The tied embedding / head and the norms stay on a fused torch AdamW (as in the FORGE quickstart)
    torch.manual_seed(seed)                                            # same init for all four optimizers
    model, params = Llama().to(DEVICE, DTYPE), None
    model.head.weight = model.embed.weight                             # tie again : .to() a TPU makes a new Parameter for each module, so the tie breaks
    if name.startswith("forge"):
        mode = name.split("_")[1]
        for parent in list(model.layers.modules()):
            for child_name, child in parent.named_children():
                if isinstance(child, nn.Linear): setattr(parent, child_name, FusedLinear.from_linear(child, state_mode = mode, lr = LR, weight_decay = WD))
        fused = {id(m.weight) for m in model.modules() if isinstance(m, FusedLinear)}
        params = [p for p in model.parameters() if id(p) not in fused]
    return model, torch.optim.AdamW(params or model.parameters(), lr = LR, weight_decay = WD, foreach = False, fused = name != "adamw")


def loss_fn(out, y): return F.cross_entropy(out.flatten(0, 1).float(), y.flatten())


if __name__ == "__main__":
    cli("llama", make_model, make_data, loss_fn, lambda params: (6 * params + 12 * LAYERS * SEQ * D) * BATCH * SEQ,   # 6 N per token (tied matrix once) + attention
        f"Llama-3.2-1B architecture, {{params}}B params (random init), batch {BATCH}, sequence {SEQ}", "Loss (cross-entropy)")
