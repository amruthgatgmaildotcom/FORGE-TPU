import torch
from forge_tpu.module import FusedLinear

#   Readme.md
#     Muon, as in the FORGE paper (2.2, App. A.2, App. I.1). For each weight matrix :   B = momentum * B + G (heavy-ball),   O = NewtonSchulz5(B),   W = W (1 - lr wd) - lr s O
#     Only B = momentum * B + G is tile-local, so FORGE fuses only that into the dW tile (kernel/optimizers.py, class Muon). Newton-Schulz needs the WHOLE matrix B, so it cannot
#     happen tile by tile : MuonStep runs it after backward, over the stored momentum. The baseline, Muon (the paper's "standard Muon"), gets G from autograd and folds it into B
#     in its own pass, then runs the SAME newton_schulz and muon_update : the two arms differ only in WHERE B = momentum * B + G happens.
#     Not in the paper, so taken from Keller Jordan's Muon (= torch.optim.Muon) : the quintic coefficients, the bf16 iteration, the transpose rule, and the update scale
#     s = sqrt(max(1, rows / cols)) (torch.optim.Muon's adjust_lr_fn "original"). Nesterov is not used : FORGE never keeps the raw G.
#     Only 2D weights take Muon. Embeddings, lm_head and norms stay on AdamW (Steps joins the optimizers of one model into one object with step / zero_grad).
#
#     This file has 3 parts :
#          1.  the math              |   newton_schulz, scale, muon_update (one matrix)
#          2.  the two optimizers    |   apply_muon (all matrices), MuonStep (FORGE Muon), Muon (standard Muon, the baseline)
#          3.  Steps                 |   several optimizers of one model as one object
#     kaggle/tp.py replaces apply_muon (the apply argument) with a tensor-parallel version, nothing else changes

NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)                                                                                           # (a, b, c) of the quintic Newton-Schulz step (Keller Jordan's Muon)



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def newton_schulz(B, steps = 5, eps = 1e-7):
    # the zeroth power (orthogonalization) of B, in bf16. The same iteration as torch.optim.Muon's _zeropower_via_newtonschulz
    a, b, c = NS_COEFFICIENTS
    X = B.to(torch.bfloat16)                                                                                                          # Newton-Schulz runs in bf16 by design
    tall = X.size(0) > X.size(1)                                                                                                      # X @ X.T is then the smaller of the two products
    if tall: X = X.T
    X = X / X.norm().clamp(min = eps)                                                                                                 # spectral norm <= 1
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X                                                                                         # X <- a X + b (X X^T) X + c (X X^T)^2 X
    return X.T if tall else X                                                                                                         # undo the transpose

def scale(shape):
    # s = sqrt(max(1, rows / cols)), the update scale of torch.optim.Muon (adjust_lr_fn "original")
    rows, cols = shape
    return max(1.0, rows / cols) ** 0.5

@torch.no_grad()
def muon_update(w, O, lr, weight_decay, full_shape = None):
    # W = W (1 - lr wd) - lr s O, in place. full_shape : the shape of the full matrix when w is a shard (tensor parallel)
    w.copy_((w.float() * (1 - lr * weight_decay) - (lr * scale(full_shape or w.shape)) * O.float()).to(w.dtype))                      # fp32 math, rounded once (as the fused kernels)



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def apply_muon(items):
    # items : [(w, B, lr, weight_decay, ns_steps)], one per matrix. O = NewtonSchulz(B), then the update of w. Kept apart so a tensor-parallel version can replace it
    for w, B, lr, weight_decay, ns_steps in items: muon_update(w, newton_schulz(B, ns_steps), lr, weight_decay)


class MuonStep:
    # FORGE Muon : the momentum is already updated inside backward (the fused kernel). step() = Newton-Schulz + the weight update of every FORGE Muon layer
    def __init__(self, module, apply = apply_muon):
        self.layers = [m for m in module.modules() if isinstance(m, FusedLinear) and m.fusedStateObject.optimizer_type == "muon"]     # every FORGE Muon layer of the module
        self.apply, self.param_groups, self.state = apply, [], {}                                                                     # no autograd gradients, no torch state (the momentum is in each layer's FusedOptimizerState)

    @torch.no_grad()
    def step(self):
        states = [layer.fusedStateObject for layer in self.layers if layer.fusedStateObject.buffers is not None]                      # buffers None : never trained
        self.apply([(s.weight, s.m, s.hparams["lr"], s.hparams["weight_decay"], s.hparams["ns_steps"]) for s in states])              # s.m : the momentum B, updated by the kernel in backward

    def zero_grad(self, set_to_none = True): pass                                                                                     # nothing to zero : FORGE layers never get a .grad


class Muon(torch.optim.Optimizer):
    # the baseline ("standard Muon" in the FORGE paper) : G from autograd, B = momentum * B + G in a separate pass, then the same Newton-Schulz and update
    def __init__(self, params, lr = 0.02, momentum = 0.95, weight_decay = 0.01, ns_steps = 5, apply = apply_muon):
        super().__init__(params, dict(lr = lr, momentum = momentum, weight_decay = weight_decay, ns_steps = ns_steps))                # the defaults of FORGE Muon (kernel/optimizers.py)
        self.apply = apply
        for group in self.param_groups:
            for p in group["params"]:
                if p.dim() != 2: raise ValueError(f"Muon takes 2D weights only, got a {tuple(p.shape)} parameter")                    # Newton-Schulz is defined for matrices only

    @torch.no_grad()
    def step(self):
        items = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.state[p]
                if "momentum_buffer" not in state: state["momentum_buffer"] = torch.zeros_like(p)                                     # the parameter's dtype (bf16 here), as FORGE's bf16 momentum
                B = state["momentum_buffer"]
                B.copy_((group["momentum"] * B.float() + p.grad.float()).to(B.dtype))                                                 # fp32 math, rounded once, as the fused kernel does
                items.append((p, B, group["lr"], group["weight_decay"], group["ns_steps"]))
        self.apply(items)                                                                                                             # the same Newton-Schulz and update as MuonStep



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



class Steps:
    # several optimizers of one model (Muon for the 2D weights, AdamW for the rest) as one object : step, zero_grad, state, param_groups
    def __init__(self, *optimizers): self.optimizers = [o for o in optimizers if o is not None]                                       # None : an optimizer with nothing to update

    def step(self):
        for o in self.optimizers: o.step()

    def zero_grad(self, set_to_none = True):
        for o in self.optimizers: o.zero_grad(set_to_none = set_to_none)

    @property
    def state(self): return {k: v for o in self.optimizers for k, v in o.state.items()}                                               # all the torch states (benchmarks/mlp.py counts their bytes)

    @property
    def param_groups(self): return [g for o in self.optimizers for g in o.param_groups]                                               # all the param groups (the benchmarks set their lr)
