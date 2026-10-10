"""Compare trained integer models (possibly with different vocabularies) on the SAME prompts.

Prompts: the first 2 characters of held-out sentences (teacher val + Wikipedia val of data/mix),
kept only when both characters exist in every model's vocabulary. For each model, with the integer
spec (t44k/spec.py, exactly what the game runs), greedy and with the decoding rule (presence
penalty 2x the model's median top-1/top-2 gap + no punctuation after punctuation):
  normal-end rate, mean length, repeat-2gram, distinct characters, and
  judge NLL: mean negative log-likelihood per generated character under the float teaching
  assistant t44k/ta_m (lower = more plausible text). Biased towards models distilled from that
  same TA; read it together with the samples.

  python -m t44k.compare_models [n_prompts]
"""
import sys, os, json, random
import numpy as np
import torch
import torch.nn.functional as F
from t44k import spec as S
from t44k.train_ta import load_ta
UNK = 3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIX = os.path.join(ROOT, "data", "mix")
MODELS = [  # name, q.json, vocab.json
    ("wiki-T1M (now in game)", "t44k/model_t1m/q.json", "data/wiki/vocab.json"),
    ("teacher-T1M", "t44k/model_t1m_tc/q.json", "data/mix/vocab.json"),
    ("teacher+KD-T1M", "t44k/model_t1m_kd/q.json", "data/mix/vocab.json"),
    ("teacher-T1M ctx64", "t44k/model_t1m64/q.json", "data/mix/vocab.json"),
    ("teacher-T4M ctx64", "t44k/model_t4m/q.json", "data/mix/vocab.json"),
    ("T4M v2 (70%, 2x data)", "t44k/model_t4m_v2/q.json", "data/mix/vocab.json"),
]
if len(sys.argv) > 2:                                   # optional: only models whose name contains this
    MODELS = [m for m in MODELS if any(k in m[0] for k in sys.argv[2].split(","))]
PUNCT = "，。！？、"


def metrics(gens):
    ends = sum(1 for g in gens if g.endswith("<E>"))
    gs = [g.replace("<E>", "") for g in gens]
    rep, dist = [], []
    for g in gs:
        bg = [g[i:i + 2] for i in range(len(g) - 1)]
        seen, r = set(), 0
        for b in bg:
            r += b in seen; seen.add(b)
        rep.append(r / max(1, len(bg))); dist.append(len(set(g)) / max(1, len(g)))
    return ends / len(gens), float(np.mean([len(g) for g in gs])), float(np.mean(rep)), float(np.mean(dist))


def main(n=200):
    S.configure("t1m")
    models = [(nm, json.load(open(os.path.join(ROOT, q))), json.load(open(os.path.join(ROOT, v), encoding="utf-8"))["vocab"])
              for nm, q, v in MODELS if os.path.exists(os.path.join(ROOT, q))]
    mix_vocab = json.load(open(os.path.join(MIX, "vocab.json"), encoding="utf-8"))["vocab"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    judge = load_ta(os.path.join(ROOT, "t44k", "ta_m", "ta.pt"), mix_vocab, dev)
    jidx = {c: i for i, c in enumerate(mix_vocab)}
    common = set.intersection(*[set(v[4:]) for _, _, v in models]) - set(PUNCT)
    src = []
    for f in ("val.txt", "aux_val.txt"):
        lines = [l.strip() for l in open(os.path.join(MIX, f), encoding="utf-8") if len(l.strip()) > 4]
        random.Random(1).shuffle(lines)
        src.append([l[:2] for l in lines if l[0] in common and l[1] in common][:n // 2])
    prompts = src[0] + src[1]
    print(f"{len(prompts)} prompts ({len(src[0])} from teacher val, {len(src[1])} from Wikipedia val)")

    def judge_nll(prompt, cont):
        ids = [S.BOS] + [jidx.get(c, UNK) for c in prompt + cont.replace("<E>", "")]
        tgt = ids[1:] + ([S.EOS] if cont.endswith("<E>") else [])
        TJ = judge.cfg["T"]                              # the judge's own context (32): scores the first part
        ids = ids[:TJ]; tgt = tgt[:TJ]
        with torch.no_grad():
            lp = F.log_softmax(judge(torch.tensor([ids], device=dev)).float()[0], -1)
        nll = [-lp[i, t].item() for i, t in enumerate(tgt) if i >= len(prompt) and t != UNK]
        return sum(nll) / max(1, len(nll))

    samples = {}
    for nm, q, vocab in models:
        S.configure(q["config"])                         # context 32 or 64
        idx = {c: i for i, c in enumerate(vocab)}
        im = S.IntModel(q)
        gaps = []
        for p in prompts[:40]:
            cache = im.new_cache()
            for pos, t in enumerate([S.BOS] + [idx[c] for c in p]):
                _, lg = im.step(t, pos, cache)
                v = np.sort(np.asarray(lg)[q["allowed"]]); gaps.append(int(v[-1] - v[-2]))
        pen = 2 * (int(np.median(gaps)) or 1)
        exempt = {idx[c] for c in PUNCT if c in idx} | {S.EOS}
        word = lambda ids: "".join("<E>" if t == S.EOS else vocab[t] for t in ids)
        for mode, kw in (("greedy", dict(penalty=0, exempt=set())), (f"rule (penalty {pen})", dict(penalty=pen, exempt=exempt))):
            gens = [word(im.generate([S.BOS] + [idx[c] for c in p], **kw)) for p in prompts]
            e, l, r, d = metrics(gens)
            j = float(np.mean([judge_nll(p, g) for p, g in zip(prompts, gens)]))
            print(f"{nm:24s} {mode:20s} normal-end {e:.2f}  len {l:4.1f}  repeat {r:.3f}  distinct {d:.2f}  judge NLL {j:.2f}", flush=True)
            samples[(nm, mode)] = gens
    print()
    for i in list(range(5)) + list(range(len(src[0]), len(src[0]) + 5)):
        print(prompts[i])
        for nm, *_ in models:
            print(f"   {nm:24s}", prompts[i] + "→" + samples[(nm, [m for (n2, m) in samples if n2 == nm][1])][i])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
