import jax.numpy as jnp

#   Readme.md
#     This file has one purpose, it is exclusively to maintain all classes of optimizers. when i am writing this the file has only one class, AdamW. To write another class in future ....
#          1.  @staticmethod def scalars(hparams, step)   |   hparams here is a {"string" : value, ...}. step is the current training step no. return a list of values only
#          2.  @staticmethod def update(w, g, st, s)      |   here st is {"state_var": matrix} is decoded state, s is scalars (in SMEM), better understood by reading AdamW.update
#          3.  static members : name (string), states (tuple of state names), floors ({state : smallest decoded value}, int8 states only), defaults (hyperparameters)
#          4.  Finally there is a dictionary at the very end of this file named OPTIMIZERS, update the new class over there also





















#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



class AdamW:

    name, states, floors = "adamw", ("m", "v"), {"v": 1e-8}                                                                            # floor only for "v" (as Dikshant, 1e-8), cuz in the update we
    defaults = {'lr': 1e-4, 'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8, 'weight_decay': 0.01}                                          # divide by sqrt(v) and int8 can round a small v to 0. bf16 states have no floor
    # these are static members                                                                                                        #

    @staticmethod
    def scalars(hp, step):                                                                                                       
        # Host side : Hyperparameters reach the kernel as runtime data, so a new lr never recompiles.                                 #
        return [hp["lr"], hp["beta1"], hp["beta2"], hp["eps"], hp["weight_decay"], 1 - hp["beta1"] ** step, 1 - hp["beta2"] ** step,
                1 - hp["beta1"], 1 - hp["beta2"]]                                                                                     # 1 - beta in double: in fp32, 1 - 0.999 is 1.3e-5 off

    @staticmethod
    def update(w, g, state_dict, scalars):
        lr, b1, b2, eps, wd, bc1, bc2, one_minus_b1, one_minus_b2 = (scalars[i] for i in range(9))                                    # w, g and st are fp32 tiles
        m = b1 * state_dict["m"] + one_minus_b1 * g                                                                                   # scalars is ref<SMEM>()
        v = b2 * state_dict["v"] + one_minus_b2 * g * g                                                                               #
        w = w * (1 - lr * wd)                                                                                                         #
        w = w - lr * (m / bc1) / (jnp.sqrt(v / bc2) + eps)                                                                            #
        return w, {"m": m, "v": v}                                                                                                    #



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#






OPTIMIZERS = {"adamw": AdamW}
