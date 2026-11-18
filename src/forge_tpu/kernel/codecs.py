import jax.numpy as jnp
from jax.experimental import pallas as pl

#   Readme.md
#     This file has one purpose, it is exclusively to maintain classes of how we want to store state variables. when i am writing this the file has two of them, int8 and Plain.
#     Plain stores each state variable in bf16 form, int8 stores it as int8 + fp32 (exact concept can be checked on internet). To write another class in future ....
#          1.  @staticmethod def arrays(V, H, dtype)      |   the arrays of ONE state variable : return a {"s1": ((size_x, size_y), dtype, init_fill), ...}
#          2.  @staticmethod def check(V, bv)             |   raise errors if needed
#          3.  @staticmethod def specs(bv, bh)            |   their blocks, same keys : return a {"s1": pl.BlockSpec((bv, bh), lambda mapping), ...}
#          4.  @staticmethod def decode(refs, floor)      |   refs is {"s1": ref, ...} of one tile, return the state tile in fp32 (floor : see optimizers.py). better understood by reading int8.decode
#          5.  @staticmethod def encode(state, out)       |   the reverse, writes the fp32 tile into out {"s1": ref, ...}. better understood by reading int8.encode   ....   hence the name of the file, encoding-decoding => codecs.py
#          6.  Finally there is a dictionary at the very end of this file named CODECS, update the new class over there also
#   
















#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



class Plain:
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    @staticmethod
    def arrays(V, H, dtype): return {"data": ((V, H), jnp.dtype(jnp.bfloat16), 0.0)}                            # state_mode "bf16": bf16 states, whatever the weights dtype
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    @staticmethod
    def check(V, bv): pass                                                                           
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    @staticmethod
    def specs(bv, bh): return {"data": pl.BlockSpec((bv, bh), lambda i, j, *_: (i, j))}
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    @staticmethod
    def decode(refs, floor): return refs["data"][...].astype(jnp.float32)
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    @staticmethod
    def encode(x, out): out["data"][...] = x.astype(out["data"].dtype)
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



class int8:
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    @staticmethod
    def arrays(V, H, dtype): return {"q": ((V, H), jnp.dtype(jnp.int8), 0), "scale": ((V//64, H), jnp.dtype(jnp.float32), 1.0)}
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    @staticmethod
    def check(V, bv):
        if V % 64: raise ValueError(f"int8 state needs out_features % 64 == 0, got {V}")
        if bv != V and bv % 512: raise ValueError(f"int8 state needs bv == {V} or a multiple of 512 (the scale block has bv // 64 rows, which must be a multiple of 8), got bv={bv}")
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    @staticmethod
    def specs(bv, bh): return {"q": pl.BlockSpec((bv, bh), lambda i, j, *_: (i, j)), "scale": pl.BlockSpec((bv//64, bh), lambda i, j, *_: (i, j))}
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    @staticmethod
    def decode(refs, floor):
        q = refs["q"][...].astype(jnp.float32)
        bv, bh = q.shape
        state = (q.reshape(bv//64, 64, bh) * refs["scale"][...][:, None, :]).reshape(bv, bh)
        return state if floor is None else jnp.maximum(state, floor)
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    @staticmethod
    def encode(state, out):
        bv, bh = state.shape; state_reshaped = state.reshape(bv//64, 64, bh)
        scale = jnp.max(jnp.abs(state_reshaped), axis = 1) / 127 + 1e-12
        inverse = 1 / scale                                                                                   # one divide per (64-row group, column), not one per element
        q = jnp.clip(jnp.round(state_reshaped * inverse[:, None, :]), -127, 127)
        out["q"][...] = q.reshape(bv, bh).astype(jnp.int8)
        out["scale"][...] = scale
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#






CODECS = {"bf16": Plain, "int8": int8}