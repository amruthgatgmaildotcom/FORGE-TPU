# FORGE correctness (adamw) : 154 / 160 checks pass

| case | check | error | tolerance | result |
|---|---|--:|--:|:--:|
| module : from_linear, eval | shares weight (0 = yes) | 0 | 0 | PASS |
| module : from_linear, eval | eval out = F.linear | 0 | 0 | PASS |
| step adamw bf16 state, float32 w, 128x256x128 | y | 0.00266 | 0.005 | PASS |
| step adamw bf16 state, float32 w, 128x256x128 | dx | 0.00237 | 0.005 | PASS |
| step adamw bf16 state, float32 w, 128x256x128 | w(rel) | 1.48e-05 | 0.005 | PASS |
| step adamw bf16 state, float32 w, 128x256x128 | m | 1 | 1 | PASS |
| step adamw bf16 state, float32 w, 128x256x128 | v | 0 | 1 | PASS |
| step adamw bf16 state, float32 w, 64x512x256 | y | 0.00289 | 0.005 | PASS |
| step adamw bf16 state, float32 w, 64x512x256 | dx | 0.00244 | 0.005 | PASS |
| step adamw bf16 state, float32 w, 64x512x256 | w(rel) | 1.47e-05 | 0.005 | PASS |
| step adamw bf16 state, float32 w, 64x512x256 | m | 1 | 1 | PASS |
| step adamw bf16 state, float32 w, 64x512x256 | v | 0 | 1 | PASS |
| step adamw bf16 state, float32 w, 256x1024x512 | y | 0.00234 | 0.005 | PASS |
| step adamw bf16 state, float32 w, 256x1024x512 | dx | 0.00241 | 0.005 | PASS |
| step adamw bf16 state, float32 w, 256x1024x512 | w(rel) | 1.47e-05 | 0.005 | PASS |
| step adamw bf16 state, float32 w, 256x1024x512 | m | 1 | 1 | PASS |
| step adamw bf16 state, float32 w, 256x1024x512 | v | 1 | 1 | PASS |
| step adamw bf16 state, float32 w, 1500x1536x2048 | y | 0.00253 | 0.005 | PASS |
| step adamw bf16 state, float32 w, 1500x1536x2048 | dx | 0.00246 | 0.005 | PASS |
| step adamw bf16 state, float32 w, 1500x1536x2048 | w(rel) | 7.35e-06 | 0.005 | PASS |
| step adamw bf16 state, float32 w, 1500x1536x2048 | m | 1 | 1 | PASS |
| step adamw bf16 state, float32 w, 1500x1536x2048 | v | 1 | 1 | PASS |
| step adamw bf16 state, bfloat16 w, 128x256x128 | y | 0.00313 | 0.005 | PASS |
| step adamw bf16 state, bfloat16 w, 128x256x128 | dx | 0.00246 | 0.005 | PASS |
| step adamw bf16 state, bfloat16 w, 128x256x128 | w(ulp) | 0 | 1 | PASS |
| step adamw bf16 state, bfloat16 w, 128x256x128 | m | 1 | 1 | PASS |
| step adamw bf16 state, bfloat16 w, 128x256x128 | v | 0 | 1 | PASS |
| step adamw bf16 state, bfloat16 w, 64x512x256 | y | 0.00256 | 0.005 | PASS |
| step adamw bf16 state, bfloat16 w, 64x512x256 | dx | 0.00273 | 0.005 | PASS |
| step adamw bf16 state, bfloat16 w, 64x512x256 | w(ulp) | 0 | 1 | PASS |
| step adamw bf16 state, bfloat16 w, 64x512x256 | m | 1 | 1 | PASS |
| step adamw bf16 state, bfloat16 w, 64x512x256 | v | 0 | 1 | PASS |
| step adamw bf16 state, bfloat16 w, 256x1024x512 | y | 0.00296 | 0.005 | PASS |
| step adamw bf16 state, bfloat16 w, 256x1024x512 | dx | 0.00237 | 0.005 | PASS |
| step adamw bf16 state, bfloat16 w, 256x1024x512 | w(ulp) | 0.0182 | 1 | PASS |
| step adamw bf16 state, bfloat16 w, 256x1024x512 | m | 1 | 1 | PASS |
| step adamw bf16 state, bfloat16 w, 256x1024x512 | v | 1 | 1 | PASS |
| step adamw bf16 state, bfloat16 w, 1500x1536x2048 | y | 0.00302 | 0.005 | PASS |
| step adamw bf16 state, bfloat16 w, 1500x1536x2048 | dx | 0.00309 | 0.005 | PASS |
| step adamw bf16 state, bfloat16 w, 1500x1536x2048 | w(ulp) | 1 | 1 | PASS |
| step adamw bf16 state, bfloat16 w, 1500x1536x2048 | m | 1 | 1 | PASS |
| step adamw bf16 state, bfloat16 w, 1500x1536x2048 | v | 1 | 1 | PASS |
| step adamw bf16 state, bfloat16 w, 256x1024x512, g x1e-06 | y | 0.00296 | 0.005 | PASS |
| step adamw bf16 state, bfloat16 w, 256x1024x512, g x1e-06 | dx | 0.00226 | 0.005 | PASS |
| step adamw bf16 state, bfloat16 w, 256x1024x512, g x1e-06 | w(ulp) | 0.00908 | 1 | PASS |
| step adamw bf16 state, bfloat16 w, 256x1024x512, g x1e-06 | m | 1 | 1 | PASS |
| step adamw bf16 state, bfloat16 w, 256x1024x512, g x1e-06 | v | 1 | 1 | PASS |
| step adamw int8 state, float32 w, 128x256x128 | y | 0.00266 | 0.005 | PASS |
| step adamw int8 state, float32 w, 128x256x128 | dx | 0.00237 | 0.005 | PASS |
| step adamw int8 state, float32 w, 128x256x128 | w(rel) | 0.000306 | 0.005 | PASS |
| step adamw int8 state, float32 w, 128x256x128 | m | 0.00393 | 0.05 | PASS |
| step adamw int8 state, float32 w, 128x256x128 | v | 0.0039 | 0.05 | PASS |
| step adamw int8 state, float32 w, 64x512x256 | y | 0.00289 | 0.005 | PASS |
| step adamw int8 state, float32 w, 64x512x256 | dx | 0.00244 | 0.005 | PASS |
| step adamw int8 state, float32 w, 64x512x256 | w(rel) | 2.37e-05 | 0.005 | PASS |
| step adamw int8 state, float32 w, 64x512x256 | m | 0.00385 | 0.05 | PASS |
| step adamw int8 state, float32 w, 64x512x256 | v | 0.00392 | 0.05 | PASS |
| step adamw int8 state, float32 w, 256x1024x512 | y | 0.00234 | 0.005 | PASS |
| step adamw int8 state, float32 w, 256x1024x512 | dx | 0.00241 | 0.005 | PASS |
| step adamw int8 state, float32 w, 256x1024x512 | w(rel) | 0.00065 | 0.005 | PASS |
| step adamw int8 state, float32 w, 256x1024x512 | m | 0.00389 | 0.05 | PASS |
| step adamw int8 state, float32 w, 256x1024x512 | v | 0.00382 | 0.05 | PASS |
| step adamw int8 state, float32 w, 1500x1536x2048 | y | 0.00253 | 0.005 | PASS |
| step adamw int8 state, float32 w, 1500x1536x2048 | dx | 0.00246 | 0.005 | PASS |
| step adamw int8 state, float32 w, 1500x1536x2048 | w(rel) | 0.00185 | 0.005 | PASS |
| step adamw int8 state, float32 w, 1500x1536x2048 | m | 0.00391 | 0.05 | PASS |
| step adamw int8 state, float32 w, 1500x1536x2048 | v | 0.00378 | 0.05 | PASS |
| step adamw int8 state, bfloat16 w, 128x256x128 | y | 0.00313 | 0.005 | PASS |
| step adamw int8 state, bfloat16 w, 128x256x128 | dx | 0.00246 | 0.005 | PASS |
| step adamw int8 state, bfloat16 w, 128x256x128 | w(ulp) | 0 | 1 | PASS |
| step adamw int8 state, bfloat16 w, 128x256x128 | m | 0.00393 | 0.05 | PASS |
| step adamw int8 state, bfloat16 w, 128x256x128 | v | 0.00391 | 0.05 | PASS |
| step adamw int8 state, bfloat16 w, 64x512x256 | y | 0.00256 | 0.005 | PASS |
| step adamw int8 state, bfloat16 w, 64x512x256 | dx | 0.00273 | 0.005 | PASS |
| step adamw int8 state, bfloat16 w, 64x512x256 | w(ulp) | 0 | 1 | PASS |
| step adamw int8 state, bfloat16 w, 64x512x256 | m | 0.00389 | 0.05 | PASS |
| step adamw int8 state, bfloat16 w, 64x512x256 | v | 0.00389 | 0.05 | PASS |
| step adamw int8 state, bfloat16 w, 256x1024x512 | y | 0.00296 | 0.005 | PASS |
| step adamw int8 state, bfloat16 w, 256x1024x512 | dx | 0.00237 | 0.005 | PASS |
| step adamw int8 state, bfloat16 w, 256x1024x512 | w(ulp) | 0.145 | 1 | PASS |
| step adamw int8 state, bfloat16 w, 256x1024x512 | m | 0.0039 | 0.05 | PASS |
| step adamw int8 state, bfloat16 w, 256x1024x512 | v | 0.00394 | 0.05 | PASS |
| step adamw int8 state, bfloat16 w, 1500x1536x2048 | y | 0.00302 | 0.005 | PASS |
| step adamw int8 state, bfloat16 w, 1500x1536x2048 | dx | 0.00309 | 0.005 | PASS |
| step adamw int8 state, bfloat16 w, 1500x1536x2048 | w(ulp) | 53.5 | 1 | **FAIL** |
| step adamw int8 state, bfloat16 w, 1500x1536x2048 | m | 0.00392 | 0.05 | PASS |
| step adamw int8 state, bfloat16 w, 1500x1536x2048 | v | 0.00376 | 0.05 | PASS |
| step adamw int8 state, bfloat16 w, 256x1024x512, g x1e-06 | y | 0.00296 | 0.005 | PASS |
| step adamw int8 state, bfloat16 w, 256x1024x512, g x1e-06 | dx | 0.00226 | 0.005 | PASS |
| step adamw int8 state, bfloat16 w, 256x1024x512, g x1e-06 | w(ulp) | 1.23e+05 | 1 | **FAIL** |
| step adamw int8 state, bfloat16 w, 256x1024x512, g x1e-06 | m | 0.0039 | 0.05 | PASS |
| step adamw int8 state, bfloat16 w, 256x1024x512, g x1e-06 | v | 1.02e+03 | 0.05 | **FAIL** |
| step adamw bf16 state, bfloat16 w, 64x512x256, bias | y | 0.00348 | 0.005 | PASS |
| step adamw bf16 state, bfloat16 w, 64x512x256, bias | dx | 0.00273 | 0.005 | PASS |
| step adamw bf16 state, bfloat16 w, 64x512x256, bias | db | 0.00239 | 0.005 | PASS |
| step adamw bf16 state, bfloat16 w, 64x512x256, bias | w(ulp) | 0 | 1 | PASS |
| step adamw bf16 state, bfloat16 w, 64x512x256, bias | m | 1 | 1 | PASS |
| step adamw bf16 state, bfloat16 w, 64x512x256, bias | v | 0 | 1 | PASS |
| chain adamw bf16 state, 2 layers, 10 steps (worst) | dh | 0.00333 | 0.005 | PASS |
| chain adamw bf16 state, 2 layers, 10 steps (worst) | dx | 0.00326 | 0.005 | PASS |
| chain adamw bf16 state, 2 layers, 10 steps (worst) | l2 w(ulp) | 1 | 1 | PASS |
| chain adamw bf16 state, 2 layers, 10 steps (worst) | l2 m | 1 | 1 | PASS |
| chain adamw bf16 state, 2 layers, 10 steps (worst) | l2 v | 1 | 1 | PASS |
| chain adamw bf16 state, 2 layers, 10 steps (worst) | l1 w(ulp) | 1 | 1 | PASS |
| chain adamw bf16 state, 2 layers, 10 steps (worst) | l1 m | 1 | 1 | PASS |
| chain adamw bf16 state, 2 layers, 10 steps (worst) | l1 v | 1 | 1 | PASS |
| chain adamw int8 state, 2 layers, 10 steps (worst) | dh | 0.00333 | 0.005 | PASS |
| chain adamw int8 state, 2 layers, 10 steps (worst) | dx | 0.00319 | 0.005 | PASS |
| chain adamw int8 state, 2 layers, 10 steps (worst) | l2 w(ulp) | 786 | 1 | **FAIL** |
| chain adamw int8 state, 2 layers, 10 steps (worst) | l2 m | 0.00392 | 0.05 | PASS |
| chain adamw int8 state, 2 layers, 10 steps (worst) | l2 v | 0.00391 | 0.05 | PASS |
| chain adamw int8 state, 2 layers, 10 steps (worst) | l1 w(ulp) | 240 | 1 | **FAIL** |
| chain adamw int8 state, 2 layers, 10 steps (worst) | l1 m | 0.00393 | 0.05 | PASS |
| chain adamw int8 state, 2 layers, 10 steps (worst) | l1 v | 0.00392 | 0.05 | PASS |
| model forge_bf16 vs PyTorch bf16 adamw | step 1 dw model.embed_tokens.weight | 2.22 | 4.44 | PASS |
| model forge_bf16 vs PyTorch bf16 adamw | step 1 dw model.input_layernorm.weight | 1 | 2 | PASS |
| model forge_bf16 vs PyTorch bf16 adamw | step 1 dw model.mlp.down_proj.weight | 2.22 | 4.44 | PASS |
| model forge_bf16 vs PyTorch bf16 adamw | step 1 dw model.mlp.gate_proj.weight | 2.22 | 4.44 | PASS |
| model forge_bf16 vs PyTorch bf16 adamw | step 1 dw model.mlp.up_proj.weight | 2.22 | 4.44 | PASS |
| model forge_bf16 vs PyTorch bf16 adamw | step 1 dw model.norm.weight | 1 | 2 | PASS |
| model forge_bf16 vs PyTorch bf16 adamw | step 1 dw model.post_attention_layernorm.weight | 1 | 2 | PASS |
| model forge_bf16 vs PyTorch bf16 adamw | step 1 dw model.self_attn.k_proj.weight | 2.22 | 4.44 | PASS |
| model forge_bf16 vs PyTorch bf16 adamw | step 1 dw model.self_attn.o_proj.weight | 2.22 | 4.44 | PASS |
| model forge_bf16 vs PyTorch bf16 adamw | step 1 dw model.self_attn.q_proj.weight | 2.22 | 4.44 | PASS |
| model forge_bf16 vs PyTorch bf16 adamw | step 1 dw model.self_attn.v_proj.weight | 2.22 | 4.44 | PASS |
| model forge_bf16 vs PyTorch bf16 adamw | 30 steps dw model.embed_tokens.weight | 0.988 | 1.98 | PASS |
| model forge_bf16 vs PyTorch bf16 adamw | 30 steps dw model.input_layernorm.weight | 1 | 2 | PASS |
| model forge_bf16 vs PyTorch bf16 adamw | 30 steps dw model.mlp.down_proj.weight | 0.98 | 1.96 | PASS |
| model forge_bf16 vs PyTorch bf16 adamw | 30 steps dw model.mlp.gate_proj.weight | 0.95 | 1.9 | PASS |
| model forge_bf16 vs PyTorch bf16 adamw | 30 steps dw model.mlp.up_proj.weight | 1 | 2 | PASS |
| model forge_bf16 vs PyTorch bf16 adamw | 30 steps dw model.norm.weight | 1 | 2 | PASS |
| model forge_bf16 vs PyTorch bf16 adamw | 30 steps dw model.post_attention_layernorm.weight | 1 | 2 | PASS |
| model forge_bf16 vs PyTorch bf16 adamw | 30 steps dw model.self_attn.k_proj.weight | 0.921 | 1.84 | PASS |
| model forge_bf16 vs PyTorch bf16 adamw | 30 steps dw model.self_attn.o_proj.weight | 0.995 | 1.99 | PASS |
| model forge_bf16 vs PyTorch bf16 adamw | 30 steps dw model.self_attn.q_proj.weight | 0.957 | 1.91 | PASS |
| model forge_bf16 vs PyTorch bf16 adamw | 30 steps dw model.self_attn.v_proj.weight | 0.96 | 1.92 | PASS |
| model forge_bf16 vs PyTorch bf16 adamw | max |loss - ref| | 0.215 | 0.441 | PASS |
| model forge_int8 vs PyTorch bf16 adamw | step 1 dw model.embed_tokens.weight | 2.22 | 4.44 | PASS |
| model forge_int8 vs PyTorch bf16 adamw | step 1 dw model.input_layernorm.weight | 1 | 2 | PASS |
| model forge_int8 vs PyTorch bf16 adamw | step 1 dw model.mlp.down_proj.weight | 1 | 4.44 | PASS |
| model forge_int8 vs PyTorch bf16 adamw | step 1 dw model.mlp.gate_proj.weight | 1.02 | 4.44 | PASS |
| model forge_int8 vs PyTorch bf16 adamw | step 1 dw model.mlp.up_proj.weight | 1.02 | 4.44 | PASS |
| model forge_int8 vs PyTorch bf16 adamw | step 1 dw model.norm.weight | 1 | 2 | PASS |
| model forge_int8 vs PyTorch bf16 adamw | step 1 dw model.post_attention_layernorm.weight | 1 | 2 | PASS |
| model forge_int8 vs PyTorch bf16 adamw | step 1 dw model.self_attn.k_proj.weight | 1 | 4.44 | PASS |
| model forge_int8 vs PyTorch bf16 adamw | step 1 dw model.self_attn.o_proj.weight | 1.04 | 4.44 | PASS |
| model forge_int8 vs PyTorch bf16 adamw | step 1 dw model.self_attn.q_proj.weight | 1 | 4.44 | PASS |
| model forge_int8 vs PyTorch bf16 adamw | step 1 dw model.self_attn.v_proj.weight | 1.02 | 4.44 | PASS |
| model forge_int8 vs PyTorch bf16 adamw | 30 steps dw model.embed_tokens.weight | 0.988 | 1.98 | PASS |
| model forge_int8 vs PyTorch bf16 adamw | 30 steps dw model.input_layernorm.weight | 1 | 2 | PASS |
| model forge_int8 vs PyTorch bf16 adamw | 30 steps dw model.mlp.down_proj.weight | 0.98 | 1.96 | PASS |
| model forge_int8 vs PyTorch bf16 adamw | 30 steps dw model.mlp.gate_proj.weight | 0.956 | 1.9 | PASS |
| model forge_int8 vs PyTorch bf16 adamw | 30 steps dw model.mlp.up_proj.weight | 1 | 2 | PASS |
| model forge_int8 vs PyTorch bf16 adamw | 30 steps dw model.norm.weight | 1 | 2 | PASS |
| model forge_int8 vs PyTorch bf16 adamw | 30 steps dw model.post_attention_layernorm.weight | 1 | 2 | PASS |
| model forge_int8 vs PyTorch bf16 adamw | 30 steps dw model.self_attn.k_proj.weight | 0.992 | 1.84 | PASS |
| model forge_int8 vs PyTorch bf16 adamw | 30 steps dw model.self_attn.o_proj.weight | 0.995 | 1.99 | PASS |
| model forge_int8 vs PyTorch bf16 adamw | 30 steps dw model.self_attn.q_proj.weight | 1 | 1.91 | PASS |
| model forge_int8 vs PyTorch bf16 adamw | 30 steps dw model.self_attn.v_proj.weight | 0.96 | 1.92 | PASS |
| model forge_int8 vs PyTorch bf16 adamw | max |loss - ref| | 0.447 | 0.441 | **FAIL** |
