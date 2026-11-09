import math, torch, torch.nn as nn
import torch.nn.functional as F
from forge_tpu.autograd import FusedLinearFunction
from forge_tpu.state import FusedOptimizerState
























#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



class FusedLinear(nn.Module):
    # This class will be the direct replacement of nn.Linear layer

    def __init__(self, in_features, out_features, bias = False, dtype = torch.bfloat16, state_mode = "bf16", optimizer_type = "adamw", **optimizer_kwargs):
        if dtype not in (torch.float32, torch.bfloat16): raise ValueError(f"FusedLinear stores its weight in torch.float32 or torch.bfloat16, got {dtype}")
        super().__init__()                                                                                                           #
        self.in_features, self.out_features = in_features, out_features                                                              #
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype = dtype))                                            # torch.empty reserves HBM
        self.bias = nn.Parameter(torch.empty(out_features, dtype = dtype)) if bias else None                                         #
        nn.init.kaiming_uniform_(self.weight, a = math.sqrt(5))                                                                      # same used by nn.Linear also
        bound = 1 / math.sqrt(in_features) if in_features > 0 else 0                                                                 # 
        if bias: nn.init.uniform_(self.bias, -bound, bound)                                                                          # 
        self.fusedStateObject = FusedOptimizerState(optimizer_type, state_mode, **optimizer_kwargs)                                  # this layer's optimizer (checks state_mode and the names), m and v come later

    @classmethod   # for now (the benchmarks use it) : wraps an existing nn.Linear, reusing its weight and bias Parameters (no copy)
    def from_linear(cls, linear, state_mode = "bf16", optimizer_type = "adamw", **optimizer_kwargs):
        with torch.device("meta"):                                                                                                   # don't allocate and initialize a weight that is replaced next
            fused = cls(linear.in_features, linear.out_features, linear.bias is not None, linear.weight.dtype, state_mode, optimizer_type, **optimizer_kwargs)
        fused.weight, fused.bias = linear.weight, linear.bias                                                                        # meta device to real device, no copies
        return fused

    def forward(self, x):
        if not self.training: return F.linear(x, self.weight, self.bias)                                                             # returns y = x @ self.weight.T + self.bias
        self.fusedStateObject.ensure_buffers(self.weight)                                                                            # first training forward only : m, v on the weight's final device
        self.weight.forge_fused = True                                                                                               # ForgeOptimizer.step() raises if this weight ever gets a .grad (tied)
        return FusedLinearFunction.apply(x, self.weight, self.bias, self.fusedStateObject)                                           # so each associated tensor now knows that .backward is in FusedLinearFunction
        # apply calls forward, and inside apply's ctx we can store anything, and when FusedLinearFunction.backward is called it can use whats stored in ctx, and backward also called fused_update
        # and fused_update needs to know optimizer details, and optimizer details are NOT typically found with nn.Linear, but FORGE forces our design choice to store optimizer details with FusedLinear

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, param_mode={'bf16' if self.weight.dtype == torch.bfloat16 else 'fp32'}, state_mode={self.fusedStateObject.state_mode}")



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#





## ================ module.py ================ ##
# 
# 
# This file contains class definiton of FusedLinear(nn.Module). FusedLinear is a direct replacement of nn.Linear
# The class stores layer specific optimizer paramenters, state_mode ("bf16"/"int8"), in and out_features, bias (bool)
# 
#       0. __init__(in_features, out_features, bias: bool = False, dtype = torch.bfloat16, state_mode = "bf16", optimizer_type = "adamw", **optimizer_kwargs):
#             The constructor initializes parameters exactly like how nn.Linear does.
#             Right now for simplicity, FusedLinear class allows param type to be only fp32 or bf16, the MXU cannot multiply
#             32 bit numbers (v5e-) but outputs proper fp32 which can be cut down to bf16 without duplicating (gate level).
#
#       1. FusedLinear.from_linear(nn.Linear, state_mode = "bf16", optimizer_type = "adamw", **optimizer_kwargs):   (for now, for the benchmarks)
#             This classmethod internally declares a meta FusedLinear, and then points towards the already existing 
#             data of nn.Linear and returns the class.
#             error is raised if state_mode not in ("bf16", "int8"). TPU v5e-1 only supports these both.