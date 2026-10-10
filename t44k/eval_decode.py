"""Offline comparison of decoding rules for the integer model (no game needed).

For prompts made of the first 2 characters of validation sentences:
  normal-end rate   share of generations that emit EOS within the context
  mean length       generated characters
  repeat-2gram      share of generated character bigrams that already occurred earlier in the
                    same sentence (looping measure; lower is better)
  distinct chars    distinct generated characters / generated characters
Variants: greedy, and greedy + presence penalty with several sizes (in integer logit units,
chosen relative to the median top-1/top-2 logit gap of the model).

  python -m t44k.eval_decode CONFIG [n_prompts]
"""
import sys, random
import numpy as np
from t44k import spec as S
from t44k.eval_luts import load

PUNCT = "，。！？、"


def metrics(gens):
    ends = sum(1 for g in gens if g and g[-1] == S.EOS)
    lens, rep, dist = [], [], []
    for g in gens:
        g = [t for t in g if t != S.EOS]
        lens.append(len(g))
        bg = list(zip(g, g[1:]))
        seen, r = set(), 0
        for b in bg:
            r += b in seen; seen.add(b)
        rep.append(r / max(1, len(bg)))
        dist.append(len(set(g)) / max(1, len(g)))
    return ends / len(gens), float(np.mean(lens)), float(np.mean(rep)), float(np.mean(dist))


def main(config="t1m", n=200):
    q, vocab, seqs = load(config)
    random.Random(1).shuffle(seqs)
    prompts = [s[:3] for s in seqs if len(s) > 4 and 3 not in s[1:3]][:n]
    im = S.IntModel(q)
    gaps = []
    for s in seqs[:60]:
        cache = im.new_cache()
        for pos, t in enumerate(s[:S.T]):
            _, lg = im.step(t, pos, cache)
            v = np.sort(np.asarray(lg)[q["allowed"]])
            gaps.append(int(v[-1] - v[-2]))
    g50 = int(np.median(gaps)) or 1
    print(f"median top1-top2 integer logit gap: {g50}  (p90 {int(np.percentile(gaps, 90))})")
    idx = {c: i for i, c in enumerate(vocab)}
    exempt = {idx[c] for c in PUNCT if c in idx} | {S.EOS}
    word = lambda ids: "".join("<E>" if t == S.EOS else vocab[t] for t in ids)
    mults = [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0.5, 1, 2, 4]
    for name, pen in [("greedy", 0)] + [(f"penalty {m}x gap = {int(m * g50)}", int(m * g50)) for m in mults]:
        gens = [im.generate(p, penalty=pen, exempt=exempt) for p in prompts]
        e, l, r, d = metrics(gens)
        print(f"\n{name:26s} normal-end {e:.2f}  mean len {l:5.1f}  repeat-2gram {r:.3f}  distinct {d:.2f}")
        for p, g in list(zip(prompts, gens))[:6]:
            print("   ", word(p[1:]) + "→" + word(g))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "t1m", int(a[1]) if len(a) > 1 else 200)
