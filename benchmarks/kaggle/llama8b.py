import os
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
import tp
import llama as one_chip                                                                     # benchmarks/llama.py : the same data (vocabulary, Markov chain, SEQ), LR and WD
from methods import MUON_LR, fuse, make_optimizer

#   Readme.md
#     Llama-3.1-8B (the numbers of the config.json of meta-llama/Llama-3.1-8B, without its rope_scaling, as benchmarks/llama.py), tensor parallel over the 8 chips (tp.py),
#     random init (std 0.02). Untied head, as the real 8B. As benchmarks/llama.py : FORGE takes every Linear inside the layers, the embedding, lm_head and the norms stay
#     on the fused torch optimizer. The names train.py and profiling.py need : make_data(seed), make_model(method, seed), loss_fn, flops, SETTINGS (and PARAMS, TITLE, SMALL,
#     build for run.py smoke). KAGGLE_TEST_TINY=1 : a small model with the same layout, only to test this code on 8 CPU processes

TINY = os.environ.get("KAGGLE_TEST_TINY") == "1"
CONFIG = LlamaConfig(vocab_size = 128256, hidden_size = 4096, intermediate_size = 14336, num_hidden_layers = 32, num_attention_heads = 32, num_key_value_heads = 8, head_dim = 128,
                     hidden_act = "silu", rope_theta = 500000.0, rms_norm_eps = 1e-5, max_position_embeddings = 131072, initializer_range = 0.02, tie_word_embeddings = False, use_cache = False)
SMALL = LlamaConfig(vocab_size = 2048, hidden_size = 256, intermediate_size = 1024, num_hidden_layers = 2, num_attention_heads = 16, num_key_value_heads = 8, head_dim = 64,
                    hidden_act = "silu", rope_theta = 500000.0, rms_norm_eps = 1e-5, max_position_embeddings = 4096, initializer_range = 0.02, tie_word_embeddings = False, use_cache = False)
if TINY: CONFIG = SMALL
LR, WD, SEQ, BATCH = one_chip.LR, one_chip.WD, one_chip.SEQ, one_chip.BATCH
PARAMS = tp.lm_params(CONFIG)                                                                # the FULL model, not one chip
TITLE = "Llama-3.1-8B"
SETTINGS = dict(one_chip.SETTINGS)                                                           # tokens_per_step, batch, seq, batches, lr, weight_decay : as benchmarks/llama.py
STATE = {}                                                                                   # the model of this run : loss_fn needs its vocabulary split



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def flops(_):
    c = CONFIG
    return (6 * PARAMS + 12 * c.num_hidden_layers * SEQ * c.hidden_size) * BATCH * SEQ      # as benchmarks/llama.py, with the parameters of the FULL model

def make_data(seed = 0):
    one_chip.DEVICE = tp.DEVICE
    if TINY: one_chip.V, one_chip.SEQ = CONFIG.vocab_size, 128
    return one_chip.make_data(seed)

def build(method, tp_on = True, config = None, seed = 0):
    model = tp.build_lm(LlamaForCausalLM, LlamaRotaryEmbedding, config or CONFIG, tp_on, seed)
    if method.startswith("forge"): fuse(model.model.layers, method, LR, WD)
    return model, make_optimizer(method, model, model.model.layers, LR, WD, **tp.muon_classes(tp_on))   # muon : the tensor-parallel Newton-Schulz

def make_model(method, seed = 0):
    model, opt = build(method, seed = seed)
    STATE.update(start = model.vocab_start, tp = True)
    return model, opt

def loss_fn(out, y): return tp.vocab_cross_entropy(out.logits, y, STATE["start"], STATE["tp"])
