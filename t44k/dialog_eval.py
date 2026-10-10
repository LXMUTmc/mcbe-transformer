"""Dialogue evaluation with the integer spec (exactly what the game runs): for held-out human
dialogues (first turn + <EOS> as the prompt) generate the reply (greedy, until <EOS>) and measure

  generic   share of replies that are generic: a normalised reply seen >= 200 times as a reply in
            the training dialogues, or a reply of <= 6 characters starting with 我也 (same rule as
            data/dedull_dialog.py)
  distinct  distinct replies / replies
  len       mean reply length (characters)
  ended     share of replies that end with <EOS> within 20 characters

  python -m t44k.dialog_eval MODEL_DIR [MODEL_DIR ...] [--n 300] [--show 20]
Prompts: data/dialog64d/val.txt (NOT de-dulled), the same for every model.
"""
import sys, os, re, json, random, collections
from t44k import spec as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
norm = lambda u: re.sub(r"[，。！？、]+$", "", u)


def generic_set(min_count=200):
    cnt = collections.Counter()
    for l in open(os.path.join(ROOT, "data", "dialog64d", "train.txt"), encoding="utf-8"):
        for u in l.rstrip("\n").split("|")[1:]:
            cnt[norm(u)] += 1
    return {u for u, x in cnt.items() if x >= min_count}


def penalty_of(im, q, idx, firsts):
    """2x the median top-1/top-2 integer logit gap over the first steps of a few prompts"""
    import numpy as np
    gaps = []
    for f in firsts[:30]:
        cache = im.new_cache()
        for pos, t in enumerate([S.BOS] + [idx[c] for c in f] + [S.EOS]):
            _, lg = im.step(t, pos, cache)
            v = np.sort(np.asarray(lg)[q["allowed"]]); gaps.append(int(v[-1] - v[-2]))
    return 2 * (int(np.median(gaps)) or 1)


def main(dirs, n=300, show=20, rule=False, prompts="dialog"):
    """prompts 'dialog': first turns of held-out LCCC dialogues; 'qa': held-out mined questions
    (data/qa_lccc/val.txt)"""
    vocab = json.load(open(os.path.join(ROOT, "data", "mixd", "vocab.json"), encoding="utf-8"))["vocab"]
    idx = {c: i for i, c in enumerate(vocab)}
    src = os.path.join(ROOT, "data", "qa_lccc" if prompts == "qa" else "dialog64d", "val.txt")
    val = [l.rstrip("\n") for l in open(src, encoding="utf-8") if l.strip()]
    random.Random(5).shuffle(val)
    firsts = [v.split("|")[0] for v in val if "□" not in v.split("|")[0]][:n]
    gen = generic_set()
    is_generic = lambda r: norm(r) in gen or (norm(r).startswith("我也") and len(norm(r)) <= 6)
    replies = {}
    for d in dirs:
        q = json.load(open(os.path.join(d, "q.json")))
        S.configure(q["config"])
        im = S.IntModel(q)
        kw = {}
        if rule:      # the game's decoding rule, penalty limited to the current turn
            ex = {idx[c] for c in "，。！？、"} | {S.EOS}
            kw = dict(penalty=penalty_of(im, q, idx, firsts), exempt=ex, turn_reset=True)
        out = []
        for f in firsts:
            p = [S.BOS] + [idx[c] for c in f] + [S.EOS]
            r = im.generate(p, max_new=20, **kw)
            out.append(("".join(vocab[t] for t in r if t != S.EOS), bool(r) and r[-1] == S.EOS))
        rs = [r for r, _ in out]
        name = os.path.basename(os.path.normpath(d)) + ("+rule" if rule else "")
        replies[name] = rs
        print(f"{name:14s} generic {sum(map(is_generic, rs)) / len(rs):.1%}  distinct {len(set(rs)) / len(rs):.1%}  "
              f"len {sum(map(len, rs)) / len(rs):.1f}  ended {sum(e for _, e in out) / len(out):.1%}  "
              f"most common: {collections.Counter(rs).most_common(5)}", flush=True)
    for i in range(min(show, len(firsts))):
        print("\n" + firsts[i])
        for name, rs in replies.items():
            print(f"   {name:14s} → {rs[i]}")


if __name__ == "__main__":
    a = sys.argv[1:]
    n = int(a[a.index("--n") + 1]) if "--n" in a else 300
    show = int(a[a.index("--show") + 1]) if "--show" in a else 20
    dirs = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in ("--n", "--show"))]
    main(dirs, n, show, rule="--rule" in a, prompts="qa" if "--qa" in a else "dialog")
