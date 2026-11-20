import math, torch, torch.nn as nn, torch.nn.functional as F
from config import OPTIM
from methods import MUON_LR, fuse, make_optimizer

#   Readme.md
#     MLP-1.1B : a deep residual MLP that learns a fixed random teacher, y = tanh(x A) B. Every parameter is a bias-free Linear weight, so FORGE takes all of them
#     (no torch optimizer is left). Big layers : the TPU is busy the whole step. A model file gives train.py, profiling.py and tests.py the same names :
#     make_data(seed), make_model(method, seed), loss_fn(out, y), flops(params), SETTINGS (written into every result file)

DEVICE, DTYPE = "tpu", torch.bfloat16
D, FF, LAYERS, BT, BATCHES = 4096, 16384, 8, 4096, 8                                         # width, hidden, blocks, tokens per step, fixed batches (cycled)
WD = 0.01
LR = {"adamw": 3e-4, "sgd": 1e-2, "muon": 3e-4}[OPTIM]                                       # muon : the AdamW lr of what is not a 2D layer weight (the MLP has nothing)
SETTINGS = dict(tokens_per_step = BT, batches = BATCHES, lr = LR, weight_decay = WD)



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Sequential(nn.Linear(D, FF, bias = False), nn.GELU("tanh"), nn.Linear(FF, D, bias = False)) for _ in range(LAYERS))
        self.head = nn.Linear(D, D, bias = False)

    def forward(self, x):
        for block in self.blocks: x = x + block(F.rms_norm(x, (D,)))
        return self.head(F.rms_norm(x, (D,)))

def make_data(seed = 0):
    # BATCHES fixed (x, y) pairs on DEVICE, y = tanh(x A) B : smooth, learnable, the same for every method
    g = torch.Generator().manual_seed(seed)
    A, B = (torch.randn(D, D, generator = g).div(math.sqrt(D)).to(DEVICE, DTYPE) for _ in range(2))
    xs = [torch.randn(BT, D, generator = g).to(DEVICE, DTYPE) for _ in range(BATCHES)]
    return [(x, torch.tanh(x @ A) @ B) for x in xs]

def make_model(method, seed = 0):
    torch.manual_seed(seed)                                                                  # the same init for every method
    model = MLP().to(DEVICE, DTYPE)
    if method.startswith("forge"): fuse(model, method, LR, WD)
    return model, make_optimizer(method, model, model, LR, WD)

def loss_fn(out, y): return F.mse_loss(out.float(), y.float())
def flops(params): return 6 * params * BT                                                    # 6 N per token (forward 2, backward 4)
