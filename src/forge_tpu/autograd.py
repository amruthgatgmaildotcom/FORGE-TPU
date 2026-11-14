import torch, torch.nn.functional as F 
from forge_tpu import _bridge

#   Readme.md
#     This file contains exactly one class for outsiders, FusedLinearFunction. it has nothing else for outsiders (_linear_backward is
#     internal). the class inherits from torch.autograd.Function, and we just have to define two staticmethods for it.
#     'forward' and 'backward'. more details about it under the line 'class FusedLinearFunction(torch.autograd.Function):'






















#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



class FusedLinearFunction(torch.autograd.Function):
    # torch.autograd.Function has a method .apply(*args), it creates some ctx objects and official docs cover how to use ctx, and calls 
    # FusedLinearFunction.forward(ctx, *args). it keeps a track of inputs which are tensors and requires_grad, and does autograd on them
    # later when we do loss.backward() it uses some autograd functionalities and calls FusedLinearFunction.backward(ctx, grad_out).
    # grad_out is computed by the autograd engine when we call loss.backward(), that is gradient of each variable wrt loss.
    # NOTE : we never call .forward() and .backward() explicitly

    @staticmethod
    def forward(ctx, input, weight, bias, fusedStateObject):
        ctx.save_for_backward(input, weight, bias)                                                                                    # during backward we need these params
        ctx.fusedStateObject = fusedStateObject                                                                                       # we need this too
        return F.linear(input, weight, bias)                                                                                          # returns y = input @ weight.T + bias

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors                                                                                       # 
        grad_input, grad_weight, grad_bias = _linear_backward(grad_output, input, weight, bias is not None, ctx.needs_input_grad[0], ctx.fusedStateObject)
        return grad_input, grad_weight, grad_bias, None                                                                               # 



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def _linear_backward(grad_output, x, weight, has_bias, needs_dx, fusedStateObject):
    # This function is for one backward pass, it updates in place unlike Torch which returns dw, and instead of dw we return None.
    # This function takes in (tensor, tensor, tensor, bool, bool, FusedOptimizerState). grad_out we borrow from autograd
    # we have to return in this order only : dx for next layer, dw of this layer (None), db of this layer (if not None else None)
    go2d = grad_output.reshape(-1, grad_output.shape[-1])                                                                             # go2d is (BT, V) tensor, needed if user uses transformers
    x2d = x.reshape(-1, x.shape[-1])                                                                                                  # x2d is (BT, H) tensor, needed if user uses transformers
    grad_bias = go2d.sum(0) if has_bias else None                                                                                     # grad_bias is (V) tensor, only if the layer has a bias (FORGE does not update it)
    fusedStateObject.increment_step()                                                                                                 # this layer's update count
    dx = _bridge.fused_update(go2d.contiguous(), x2d.contiguous(), fusedStateObject, needs_dx)                                        # updates w and its state, returns dx
    return None if dx is None else dx.reshape(*grad_output.shape[:-1], -1), None, grad_bias                                           # 



