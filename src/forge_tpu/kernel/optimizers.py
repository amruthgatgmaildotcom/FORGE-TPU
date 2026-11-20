import jax.numpy as jnp

#   Readme.md
#     This file has one purpose, it is exclusively to maintain all classes of optimizers. when i am writing this the file has two classes, AdamW and SGD. To write another class in future ....
#          1.  @staticmethod def scalars(hparams, step)   |   hparams here is a {"string" : value, ...}. step is the current training step no. return a list of values only
#          2.  @staticmethod def update(w, g, st, s)      |   here st is {"state_var": matrix} is decoded state, s is scalars (in SMEM), better understood by reading AdamW.update
#          3.  static members : name (string), states (tuple of state names, () : no state), floors ({state : smallest decoded value}, int8 states only), defaults (hyperparameters)
#              optional : modes (the state_modes it allows, default all of CODECS), weight (False : backward does not change W, the kernel never loads it ; default True)
#          4.  Finally there is a dictionary at the very end of this file named OPTIMIZERS, update the new class over there also





















#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



class AdamW:

    name, states, floors = "adamw", ("m", "v"), {"v": 1e-8}                                                                            # floor only for "v" (1e-8, as in FORGE), cuz in the update we
    defaults = {'lr': 1e-4, 'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8, 'weight_decay': 0.01}                                          # divide by sqrt(v) and int8 can round a small v to 0. bf16 states have no floor
    # these are static members                                                                                                        #

    @staticmethod
    def scalars(hp, step):                                                                                                       
        # Host side : Hyperparameters reach the kernel as runtime data, so a new lr never recompiles.                                 #
        bc1, bc2 = 1 - hp["beta1"] ** step, 1 - hp["beta2"] ** step                                                                   # in double on the host, as torch
        return [hp["lr"] / bc1, 1 / bc2 ** 0.5, hp["eps"], 1 - hp["lr"] * hp["weight_decay"], hp["beta1"], hp["beta2"], 1 - hp["beta1"], 1 - hp["beta2"]]  # 1 - beta in double: in fp32, 1 - 0.999 is 1.3e-5 off

    @staticmethod
    def update(w, g, state_dict, scalars):
        step_size, inv_sqrt_bc2, eps, decay, b1, b2, one_minus_b1, one_minus_b2 = (scalars[i] for i in range(8))                  # w, g and st are fp32 tiles, scalars is ref<SMEM>()
        m = b1 * state_dict["m"] + one_minus_b1 * g                                                                                  # torch.optim.AdamW (_single_tensor_adam) step by step :
        v = b2 * state_dict["v"] + one_minus_b2 * g * g                                                                              #   denom = sqrt(v) / sqrt(bc2) + eps,  w = w (1 - lr wd) - (lr / bc1) m / denom
        w = w * decay - step_size * m / (jnp.sqrt(v) * inv_sqrt_bc2 + eps)                                                         # ONE divide per element (was three : m / bc1, v / bc2, then the update)
        return w, {"m": m, "v": v}                                                                                                    #



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#






class SGD:
    
    name, states, floors = "sgd", (), {}                                                                                              # no state : no state_mode (see state.py)
    defaults = {'lr': 1e-3, 'weight_decay': 0.0}                                                                                      # the defaults of torch.optim.SGD
    # plain SGD (no momentum), as torch.optim.SGD(lr, weight_decay) with momentum = 0 : no optimizer state at all, so FORGE keeps only the weights

    @staticmethod
    def scalars(hp, step):
        return [hp["lr"], hp["weight_decay"]]                                                                                         # step is not used : plain SGD has no bias correction

    @staticmethod
    def update(w, g, state_dict, scalars):
        lr, wd = scalars[0], scalars[1]                                                                                               # w, g are fp32 tiles, state_dict is {}
        w = w - lr * (g + wd * w)                                                                                                     # as torch : g + wd * w, then w - lr * g
        return w, {}                                                                                                                  #



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#






class Muon:

    name, states, floors, modes, weight = "muon", ("m",), {}, ("bf16",), False                                                        # bf16 momentum only (FORGE paper : int8 / fp8 not on the Muon path). weight = False : backward does not change W (Newton-Schulz does, after backward)
    defaults = {'lr': 0.02, 'momentum': 0.95, 'weight_decay': 0.01, 'ns_steps': 5}                                                    # momentum 0.95, 5 Newton-Schulz steps : the FORGE paper (App. I.1)
    # Muon as in the FORGE paper (2.2, App. A.2) : only the momentum accumulate B = momentum * B + G is tile-local, so only it is fused here. Newton-Schulz needs the
    # whole matrix B : it runs after backward, over the stored momentum (forge_tpu/muon.py, MuonStep). The weight passes through this kernel unchanged.

    @staticmethod
    def scalars(hp, step):
        return [hp["momentum"]]                                                                                                       # lr, weight_decay, ns_steps : used after backward, not here

    @staticmethod
    def update(w, g, state_dict, scalars):
        return w, {"m": scalars[0] * state_dict["m"] + g}                                                                             # heavy-ball, as the paper : B <- mu B + G (no (1 - mu), no Nesterov). w is None here (weight = False : the kernel does not load it)



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#






OPTIMIZERS = {"adamw": AdamW, "sgd": SGD, "muon": Muon}
