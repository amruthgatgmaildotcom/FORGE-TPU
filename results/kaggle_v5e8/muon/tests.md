# FORGE correctness (muon) : 72 / 72 checks pass

| case | check | error | tolerance | result |
|---|---|--:|--:|:--:|
| module : from_linear, eval | shares weight (0 = yes) | 0 | 0 | PASS |
| module : from_linear, eval | eval out = F.linear | 0 | 0 | PASS |
| step muon bf16 state, float32 w, 128x256x128 | y | 0.00266 | 0.005 | PASS |
| step muon bf16 state, float32 w, 128x256x128 | dx | 0.00237 | 0.005 | PASS |
| step muon bf16 state, float32 w, 128x256x128 | w(rel) | 0.0861 | 0.1 | PASS |
| step muon bf16 state, float32 w, 128x256x128 | m | 1 | 1 | PASS |
| step muon bf16 state, float32 w, 64x512x256 | y | 0.00289 | 0.005 | PASS |
| step muon bf16 state, float32 w, 64x512x256 | dx | 0.00244 | 0.005 | PASS |
| step muon bf16 state, float32 w, 64x512x256 | w(rel) | 0.0715 | 0.1 | PASS |
| step muon bf16 state, float32 w, 64x512x256 | m | 1 | 1 | PASS |
| step muon bf16 state, float32 w, 256x1024x512 | y | 0.00234 | 0.005 | PASS |
| step muon bf16 state, float32 w, 256x1024x512 | dx | 0.00241 | 0.005 | PASS |
| step muon bf16 state, float32 w, 256x1024x512 | w(rel) | 0.0665 | 0.1 | PASS |
| step muon bf16 state, float32 w, 256x1024x512 | m | 1 | 1 | PASS |
| step muon bf16 state, float32 w, 1500x1536x2048 | y | 0.00253 | 0.005 | PASS |
| step muon bf16 state, float32 w, 1500x1536x2048 | dx | 0.00246 | 0.005 | PASS |
| step muon bf16 state, float32 w, 1500x1536x2048 | w(rel) | 0.0473 | 0.1 | PASS |
| step muon bf16 state, float32 w, 1500x1536x2048 | m | 1 | 1 | PASS |
| step muon bf16 state, bfloat16 w, 128x256x128 | y | 0.00313 | 0.005 | PASS |
| step muon bf16 state, bfloat16 w, 128x256x128 | dx | 0.00246 | 0.005 | PASS |
| step muon bf16 state, bfloat16 w, 128x256x128 | w(ulp+NS) | 0.939 | 1 | PASS |
| step muon bf16 state, bfloat16 w, 128x256x128 | m | 1 | 1 | PASS |
| step muon bf16 state, bfloat16 w, 64x512x256 | y | 0.00256 | 0.005 | PASS |
| step muon bf16 state, bfloat16 w, 64x512x256 | dx | 0.00273 | 0.005 | PASS |
| step muon bf16 state, bfloat16 w, 64x512x256 | w(ulp+NS) | 0.9 | 1 | PASS |
| step muon bf16 state, bfloat16 w, 64x512x256 | m | 1 | 1 | PASS |
| step muon bf16 state, bfloat16 w, 256x1024x512 | y | 0.00296 | 0.005 | PASS |
| step muon bf16 state, bfloat16 w, 256x1024x512 | dx | 0.00237 | 0.005 | PASS |
| step muon bf16 state, bfloat16 w, 256x1024x512 | w(ulp+NS) | 0.921 | 1 | PASS |
| step muon bf16 state, bfloat16 w, 256x1024x512 | m | 1 | 1 | PASS |
| step muon bf16 state, bfloat16 w, 1500x1536x2048 | y | 0.00302 | 0.005 | PASS |
| step muon bf16 state, bfloat16 w, 1500x1536x2048 | dx | 0.00309 | 0.005 | PASS |
| step muon bf16 state, bfloat16 w, 1500x1536x2048 | w(ulp+NS) | 0.922 | 1 | PASS |
| step muon bf16 state, bfloat16 w, 1500x1536x2048 | m | 1 | 1 | PASS |
| step muon bf16 state, bfloat16 w, 256x1024x512, g x1e-06 | y | 0.00296 | 0.005 | PASS |
| step muon bf16 state, bfloat16 w, 256x1024x512, g x1e-06 | dx | 0.00226 | 0.005 | PASS |
| step muon bf16 state, bfloat16 w, 256x1024x512, g x1e-06 | w(ulp+NS) | 0.921 | 1 | PASS |
| step muon bf16 state, bfloat16 w, 256x1024x512, g x1e-06 | m | 1 | 1 | PASS |
| step muon bf16 state, bfloat16 w, 64x512x256, bias | y | 0.00348 | 0.005 | PASS |
| step muon bf16 state, bfloat16 w, 64x512x256, bias | dx | 0.00273 | 0.005 | PASS |
| step muon bf16 state, bfloat16 w, 64x512x256, bias | db | 0.00239 | 0.005 | PASS |
| step muon bf16 state, bfloat16 w, 64x512x256, bias | w(ulp+NS) | 0.949 | 1 | PASS |
| step muon bf16 state, bfloat16 w, 64x512x256, bias | m | 1 | 1 | PASS |
| chain muon bf16 state, 2 layers, 10 steps (worst) | dh | 0.00334 | 0.005 | PASS |
| chain muon bf16 state, 2 layers, 10 steps (worst) | dx | 0.00319 | 0.005 | PASS |
| chain muon bf16 state, 2 layers, 10 steps (worst) | l2 w(ulp+NS) | 0.949 | 1 | PASS |
| chain muon bf16 state, 2 layers, 10 steps (worst) | l2 m | 1 | 1 | PASS |
| chain muon bf16 state, 2 layers, 10 steps (worst) | l1 w(ulp+NS) | 0.949 | 1 | PASS |
| chain muon bf16 state, 2 layers, 10 steps (worst) | l1 m | 1 | 1 | PASS |
| model forge_muon vs PyTorch bf16 muon | step 1 dw model.embed_tokens.weight | 2.22 | 4.44 | PASS |
| model forge_muon vs PyTorch bf16 muon | step 1 dw model.input_layernorm.weight | 1 | 2 | PASS |
| model forge_muon vs PyTorch bf16 muon | step 1 dw model.mlp.down_proj.weight | 0.295 | 0.589 | PASS |
| model forge_muon vs PyTorch bf16 muon | step 1 dw model.mlp.gate_proj.weight | 0.314 | 0.629 | PASS |
| model forge_muon vs PyTorch bf16 muon | step 1 dw model.mlp.up_proj.weight | 0.279 | 0.558 | PASS |
| model forge_muon vs PyTorch bf16 muon | step 1 dw model.norm.weight | 1 | 2 | PASS |
| model forge_muon vs PyTorch bf16 muon | step 1 dw model.post_attention_layernorm.weight | 1 | 2 | PASS |
| model forge_muon vs PyTorch bf16 muon | step 1 dw model.self_attn.k_proj.weight | 0.12 | 0.24 | PASS |
| model forge_muon vs PyTorch bf16 muon | step 1 dw model.self_attn.o_proj.weight | 0.916 | 1.83 | PASS |
| model forge_muon vs PyTorch bf16 muon | step 1 dw model.self_attn.q_proj.weight | 0.275 | 0.55 | PASS |
| model forge_muon vs PyTorch bf16 muon | step 1 dw model.self_attn.v_proj.weight | 0.578 | 1.16 | PASS |
| model forge_muon vs PyTorch bf16 muon | 30 steps dw model.embed_tokens.weight | 0.98 | 1.96 | PASS |
| model forge_muon vs PyTorch bf16 muon | 30 steps dw model.input_layernorm.weight | 1 | 2 | PASS |
| model forge_muon vs PyTorch bf16 muon | 30 steps dw model.mlp.down_proj.weight | 0.0812 | 0.181 | PASS |
| model forge_muon vs PyTorch bf16 muon | 30 steps dw model.mlp.gate_proj.weight | 0.0922 | 0.191 | PASS |
| model forge_muon vs PyTorch bf16 muon | 30 steps dw model.mlp.up_proj.weight | 0.0852 | 0.176 | PASS |
| model forge_muon vs PyTorch bf16 muon | 30 steps dw model.norm.weight | 1 | 2 | PASS |
| model forge_muon vs PyTorch bf16 muon | 30 steps dw model.post_attention_layernorm.weight | 1 | 2 | PASS |
| model forge_muon vs PyTorch bf16 muon | 30 steps dw model.self_attn.k_proj.weight | 0.085 | 0.179 | PASS |
| model forge_muon vs PyTorch bf16 muon | 30 steps dw model.self_attn.o_proj.weight | 0.336 | 0.838 | PASS |
| model forge_muon vs PyTorch bf16 muon | 30 steps dw model.self_attn.q_proj.weight | 0.118 | 0.224 | PASS |
| model forge_muon vs PyTorch bf16 muon | 30 steps dw model.self_attn.v_proj.weight | 0.184 | 0.433 | PASS |
| model forge_muon vs PyTorch bf16 muon | max |loss - ref| | 0.0491 | 0.13 | PASS |
