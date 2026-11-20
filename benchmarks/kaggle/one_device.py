import copy, glob, json, os, sys

#   Readme.md
#     python benchmarks/kaggle/one_device.py <results folder of the Kaggle run>   ->   <run>_1dev.json next to every <method>_seed<N>.json of llama8b
#     The Kaggle run splits Llama-3.1-8B over 8 chips (tensor parallel). This file converts its memory to ONE device (one HBM), so it compares with one-device training :
#        weights, optimizer states, weight gradient : 8 x one chip (each chip holds exactly 1/8 of them : the 8 x 1/8 make the full tensors)
#        activations                                : K x one chip. Most activations of a chip are NOT split (the residual stream and the norm activations are the same on every
#                                                     chip), so 8 x would count them 8 times. K = (one-device activations) / (one-chip activations) before backward, from a one-device
#                                                     run of the same model and data on another accelerator (3.6 GiB at 512 tokens)
#     Every memory point of a run (the per-layer trace, the peak of every step, the peak) is converted : at a point of the backward pass the weight gradient that exists is
#     G x (the part of the backward pass done), the rest above weights + states is activations. The converted file has the same keys, and one more : one_device = {K, from_chip_peak}

K = 2.636
SPLIT = ("weights", "states", "grads")



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def convert(r):
    c = copy.deepcopy(r)
    g = r["memory_gib"]
    W, S, G = g["weights"], g["states"], g["grads"]
    static = W + S
    trace = r["trace"]
    at = {p[2]: i for i, p in enumerate(trace) if p[2]}
    i0, i1 = at.get("loss", 0), at.get("backward", len(trace) - 1)
    def one(i, v):
        # v : memory (GiB) of one chip at trace point i -> one device
        done = 0.0 if i <= i0 else 1.0 if i >= i1 else (i - i0) / (i1 - i0)
        a = max(v - static - G * done, 0.0)
        return 8 * static + 8 * min(G * done, max(v - static, 0.0)) + K * a
    c["trace"] = [[one(i, p[0]), one(i, p[1]), p[2]] for i, p in enumerate(trace)]
    peak_chip = r.get("peak_synced_gib", g["peak"])
    peak_1 = max(p[1] for p in c["trace"])
    scale = (peak_1 - 8 * static) / max(peak_chip - static, 1e-9)                            # the timed peaks of every step : the same split as the traced peak
    c["peak_gib"] = [8 * static + (m - static) * scale for m in r["peak_gib"]]
    c["held_gib"] = [one(i1, h) for h in r["held_gib"]]
    for k in SPLIT: c["memory_gib"][k] = 8 * g[k]
    c["memory_gib"]["peak"] = peak_1
    c["memory_gib"]["activations"] = peak_1 - sum(c["memory_gib"][k] for k in SPLIT)
    c["one_device"] = dict(K = K, from_chip_peak = peak_chip)
    return c

def main(root):
    for f in sorted(glob.glob(os.path.join(root, "*", "llama8b", "*_seed*.json"))):
        if f.endswith("_1dev.json"): continue
        json.dump(convert(json.load(open(f))), open(f[:-5] + "_1dev.json", "w"))
        print(f"one device : {f[:-5]}_1dev.json")

if __name__ == "__main__": main(sys.argv[1])
