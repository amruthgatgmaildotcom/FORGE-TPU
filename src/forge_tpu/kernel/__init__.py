import jax.numpy as jnp
from ._utils import lookup
from .codecs import CODECS
from .fused_grad import fused_grad
from .optimizers import OPTIMIZERS

__all__ = ["fused_grad", "OPTIMIZERS", "CODECS", "state_arrays"]







def state_arrays(optimizer, state_mode, V, H, dtype):
    # What to allocate for one (V, H) weight: {state: {array_name: (shape, dtype, fill)}}.
    opt, codec = lookup(optimizer, state_mode)
    codec.check(V, V)
    return {state_string: codec.arrays(V, H, jnp.dtype(dtype)) for state_string in opt.states}

