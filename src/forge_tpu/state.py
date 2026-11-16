from forge_tpu import _bridge
from forge_tpu.kernel import OPTIMIZERS

#   Readme.md
#     This file has only one class to offer to outsiders, FusedOptimizerState. constructor needs to know the "optimizer_type", "state_mode", **hparams (default if not given)
#     each FusedLinear layer has its own FusedOptimizerState object, which holds everything about the optimizer specific to that layer. 
#     if we create state matrices inside the contructor of FusedLinear, and then when we do .to(device) states wont be moved to device. we could have tried initializing 
#     states as register_buffers or nn.Parameter(requires_grad = False) and worked it out, but lets modularize everything related to a layer's optimizer into one class .
#     We init states directly on the device during the first forward pass (that is what even pytorch does), and hence the method ensure_buffers (buffers is the word used
#     state matrices here)


















#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def _map(buffers, fn):                                                                                                              # recall that buffers hold state variables {str : {str : , ...}, ...}
    return None if buffers is None else {state_string: {stringg: fn(tensoR) for stringg, tensoR in state_dictionary.items()} for state_string, state_dictionary in buffers.items()}


class FusedOptimizerState:
    # One weight's optimizer : optimizer_type and state_mode (strings), hyperparameters (self.hparams : dict, read as hparams["lr"]),
    # the update count (self.step : int), the weight (self.weight) and the state tensors (self.buffers : {str : {str : tensor, }, }).
    # Made in FusedLinear.__init__ (names checked right away). ensure_buffers binds the weight and allocates the buffers at the first
    # training forward, when the weight is on its final device and dtype (after any .to()).

    def __init__(self, optimizer_type = "adamw", state_mode = "bf16", **hparams):
        if optimizer_type not in OPTIMIZERS: raise ValueError(f"unknown optimizer {optimizer_type!r}, available: {sorted(OPTIMIZERS)}")
        if state_mode not in ("bf16", "int8"): raise ValueError(f"unknown state_mode {state_mode!r}, use 'bf16' (plain) or 'int8'") #
        defaults = OPTIMIZERS[optimizer_type].defaults                                                                              # 
        if hparams.keys() - defaults.keys():                                                                                        # if there exists an unknown param
            raise TypeError(f"unknown {optimizer_type} hyperparameters {sorted(hparams.keys() - defaults.keys())}, expected some of {sorted(defaults)}")
        self.optimizer_type, self.state_mode = optimizer_type, state_mode                                                           # 
        self.hparams = {**defaults, **hparams}                                                                                      # merging dictionaries, values of hparams over writes defaults if keys clash
        self.step, self.weight, self.buffers = 0, None, None                                                                        # weight and buffers : see ensure_buffers

    def ensure_buffers(self, weight):
        # binds the weight and allocates the space on HBM for the optimizer states (once), self.buffers become {str : {str : tensor, }, }
        # Called at every training forward : the module can get a NEW Parameter after the first one (.to() a TPU, load_state_dict(assign = True), ...).
        # Then the update must go to the new Parameter, else the module's weight never changes (the old one is updated, silently). The states are kept
        # (moved to the new weight's device if it changed), the shape must be the same
        if self.buffers is None:
            self.weight, self.buffers = weight, _bridge.allocate_state(weight, self.optimizer_type, self.state_mode)
        elif weight is not self.weight:
            if weight.shape != self.weight.shape: raise RuntimeError(f"FusedLinear got a new weight of shape {tuple(weight.shape)}, its optimizer states are for {tuple(self.weight.shape)}")
            if weight.device != self.weight.device: self.buffers = _map(self.buffers, lambda tensoR: tensoR.to(weight.device))
            self.weight = weight

    def increment_step(self):                                                                                                       #
        # we also maintain how many times this layer has been updated (AdamW needs it for the bias correction, see optimizers.py)
        self.step += 1

    def state_dict(self):                                                                                                           # how to save state, we need to copy bacck to host to save on disk
        return {"optimizer_type": self.optimizer_type, "state_mode": self.state_mode, "step": self.step, "buffers": _map(self.buffers, lambda tensoR: tensoR.to("cpu", copy = True))}

    def load_state_dict(self, d):                                                       
        # call ensure_buffers(layer.weight) first : the weight decides the device the buffers go to
        if self.weight is None: raise RuntimeError("bind the weight first: fusedStateObject.ensure_buffers(layer.weight), then load_state_dict(d)")
        if (d["optimizer_type"], d["state_mode"]) != (self.optimizer_type, self.state_mode): raise ValueError(f"state is for ({d['optimizer_type']}, {d['state_mode']}), this layer uses ({self.optimizer_type}, {self.state_mode})")
        self.step = d["step"]
        # Through the host (as in state_dict) : the layer must never share a buffer with the caller's tensors, the update gives its buffers away (donation, see _bridge._rebind)
        self.buffers = _map(d["buffers"], lambda tensoR: tensoR.to("cpu", copy = True).to(self.weight.device))

    def __getattr__(self, name):
        # Read-only views of the state arrays, for any optimizer (None when absent): for adamw state.m, state.v (plain), state.m_q, state.m_scale, ... (int8). Python calls this only for names that are not normal attributes.
        opt = OPTIMIZERS.get(self.__dict__.get("optimizer_type"))                                                                   # __dict__ : no recursion
        state, array = (name, "data") if opt and name in opt.states else name.rpartition("_")[::2]                                  # "m" -> (m, data), "m_q" -> (m, q)
        if opt is None or state not in opt.states: raise AttributeError(name)                                                       # a typo stays an error
        return (self.__dict__.get("buffers") or {}).get(state, {}).get(array)                                                       #



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
