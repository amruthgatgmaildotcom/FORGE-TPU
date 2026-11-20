import torch, torch.nn as nn
from forge_tpu import FusedLinear, Muon, MuonStep, Steps
from config import OPTIM, WARMUP

#   Readme.md
#     How each method trains : which weights FORGE takes (fuse), which torch optimizer takes the rest (make_optimizer), and the lr warmup (lr_warmup).
#     The models (mlp.py, llama.py, qwen.py) call fuse and make_optimizer in their make_model; train.py calls lr_warmup.
#        vanilla torch   torch's optimizer with its defaults (foreach), for every parameter
#        fused torch     the same with fused = True
#        FORGE           every nn.Linear inside `scope` becomes a FusedLinear (its update runs inside backward); the parameters left over (embedding, norms) go to the FUSED torch optimizer
#        Muon            Muon on the 2D weights of the Linears inside `scope`, fused AdamW on the rest. FORGE (Muon) : MuonStep for the FORGE layers, the same fused AdamW on the rest

MUON_LR = 0.02                                                                               # Muon's lr (Keller Jordan's Muon, torch.optim.Muon's default); the AdamW lr of a model is for the rest



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def fuse(scope, method, lr, wd):
    # every nn.Linear inside scope -> a FusedLinear that shares its weight (no copy)
    kwargs = {"forge_bf16": dict(optimizer_type = "adamw", state_mode = "bf16", lr = lr), "forge_int8": dict(optimizer_type = "adamw", state_mode = "int8", lr = lr),
              "forge_sgd":  dict(optimizer_type = "sgd", lr = lr),                   "forge_muon": dict(optimizer_type = "muon", lr = MUON_LR)}[method]
    for parent in list(scope.modules()):
        for name, child in parent.named_children():
            if isinstance(child, nn.Linear): setattr(parent, name, FusedLinear.from_linear(child, weight_decay = wd, **kwargs))

def torch_optimizer(method, params, lr, wd):
    # vanilla : torch's defaults. fused = False is NOT passed : torch would then also turn its default foreach off. A FORGE method gets the fused one for what it does not fuse
    fused = {"fused": True} if method.endswith("_fused") or method.startswith("forge") else {}
    return {"adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}[OPTIM](params, lr = lr, weight_decay = wd, **fused)

def make_optimizer(method, model, scope, lr, wd, muon = Muon, muon_step = MuonStep):
    # the optimizer of a method, after fuse() for a FORGE method. None : FORGE updates every parameter (the MLP), there is nothing left for torch
    fused = {id(m.weight) for m in model.modules() if isinstance(m, FusedLinear)}
    if OPTIM != "muon":
        params = [p for p in model.parameters() if id(p) not in fused]
        return torch_optimizer(method, params, lr, wd) if params else None
    if method == "forge_muon": layers, rest = muon_step(model), [p for p in model.parameters() if id(p) not in fused]
    else:
        weights = [m.weight for m in scope.modules() if isinstance(m, nn.Linear)]
        layers = muon(weights, lr = MUON_LR, weight_decay = wd)
        rest = [p for p in model.parameters() if id(p) not in {id(w) for w in weights}]
    return Steps(layers, torch.optim.AdamW(rest, lr = lr, weight_decay = wd, fused = True) if rest else None)



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def lr_warmup(model, opt):
    # returns set_lr(step) : every lr of the method (its torch param groups and every FORGE layer) = its base lr x min(1, (step + 1) / WARMUP). The same for every method.
    # A random-init model at the full lr from step 0 makes loss spikes. FORGE reads its lr as runtime data (no recompile), and the WARMUP steps are not timed
    targets = [(d, d["lr"]) for d in (opt.param_groups if opt else []) + [m.fusedStateObject.hparams for m in model.modules() if isinstance(m, FusedLinear)]]
    def set_lr(step):
        if step > WARMUP: return
        for d, base in targets: d["lr"] = base * min(1.0, (step + 1) / WARMUP)
    return set_lr
