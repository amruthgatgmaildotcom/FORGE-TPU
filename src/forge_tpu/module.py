import math, torch, warnings, torch.nn as nn
import torch.nn.functional as F
from forge_tpu.autograd import FusedLinearFunction
from forge_tpu.state import FusedOptimizerState

#   Readme.md 
#      when we say loss.backward() it triggers some tensors, autograd and nn.Linear ultimately. so to achieve FORGE we must make our own nn.Linear (call it FusedLinear)
#      This layer class works exactly like nn.Linear but it also needs the optimizer details (this is my design choice).
#      it also has a classmethod called from_linear which steals/shares with already existing nn.Linear object and modifies it to be a FusedLinear
#      while i am writing this, for simplicity only torch.bfloat16 and torch.float32 are allowed for parameters and v5e only does math on max 16 bit numbers natively
#      although 32 bit floats can be broken into 2 and used for states, right now only state types allowed are bf16 and int8 (+ fp32 quantized)


















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
        if bias: warnings.warn("FusedLinear(bias = True) : FORGE updates the weight only. The bias gets a .grad in backward, so give it to a torch optimizer (as benchmarks/llama.py does for its non-fused params), or it never changes")
        self.fusedStateObject = FusedOptimizerState(optimizer_type, state_mode, **optimizer_kwargs)                                  # this layer's optimizer (checks state_mode and the names), m and v come later

    @classmethod   # for now (the benchmarks use it) : wraps an existing nn.Linear, reusing its weight and bias Parameters (no copy)
    def from_linear(cls, linear, state_mode = "bf16", optimizer_type = "adamw", **optimizer_kwargs):
        with torch.device("meta"):                                                                                                   # don't allocate and initialize a weight that is replaced next
            fused = cls(linear.in_features, linear.out_features, linear.bias is not None, linear.weight.dtype, state_mode, optimizer_type, **optimizer_kwargs)
        fused.weight, fused.bias = linear.weight, linear.bias                                                                        # meta device to real device, no copies
        return fused

    def forward(self, x):
        if not self.training: return F.linear(x, self.weight, self.bias)                                                             # returns y = x @ self.weight.T + self.bias
        self.fusedStateObject.ensure_buffers(self.weight)                                                                            # m, v at the first training forward, the weight bound again if it is a new Parameter
        return FusedLinearFunction.apply(x, self.weight, self.bias, self.fusedStateObject)                                           # so each associated tensor now knows that .backward is in FusedLinearFunction
        # apply calls forward, and inside apply's ctx we can store anything, and when FusedLinearFunction.backward is called it can use whats stored in ctx, and backward also called fused_update
        # and fused_update needs to know optimizer details, and optimizer details are NOT typically found with nn.Linear, but FORGE forces our design choice to store optimizer details with FusedLinear

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, param_mode={'bf16' if self.weight.dtype == torch.bfloat16 else 'fp32'}, state_mode={self.fusedStateObject.state_mode}")



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



