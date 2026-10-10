"""Offline test: does an n-gram table (the kind that could live in BLOCK MEMORY in the game) help
the integer model?

Counts over the model's own training text (teacher, Wikipedia, dialogues + mined QA; <BOS> line
<EOS>, '|' = <EOS>):
  tri  : (prev2, prev1) -> next, direct address prev2*V + prev1 (V*V = 1M cells, no hash)
  four : (prev3, prev2, prev1) -> next, hashed into --slots cells (collisions as in the game)
Each context keeps its top-K next tokens with counts (what fits a block-memory row).
Combination (log-linear, renormalised over the allowed tokens):
  score_t = s * z_t + w * log(p_ngram(t) + eps)     z = integer logits of the spec model
  p_ngram(t) = count_t / total  for t in the top-K list, 0 otherwise (-> eps)
  four-gram first, tri-gram when the four-gram context was seen fewer than --min4 times
s and w are fitted on half of the validation positions and reported on the other half.

  python -m t44k.ngram_eval [model_dir] [--k 4] [--seqs 300]
"""
import sys, os, json, random, time
import numpy as np
from t44k import spec as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = ["data/mix64d/train.txt", "data/mix64d/aux_train.txt", "data/dialog64f/train.txt"]


def encode_lines(paths, idx, limit=None):
    seqs = []
    for p in paths:
        for k, l in enumerate(open(os.path.join(ROOT, p), encoding="utf-8")):
            l = l.rstrip("\n")
            if not l:
                continue
            ids = [S.BOS] + [S.EOS if c == "|" else idx.get(c, 3) for c in l][:S.T - 1]
            if not l.endswith("，"):
                ids.append(S.EOS)
            seqs.append(ids)
            if limit and len(seqs) >= limit:
                break
    return seqs


def encode_flat(paths, idx):
    """all training text as ONE int16 token array: <BOS> line <EOS> <BOS> line <EOS> ... ('|' = <EOS>)
    (numpy, ~2 bytes per token; every line counts as ended, the few comma-cut lines included)"""
    lut = np.full(0x10000, 3, np.int16)                  # unknown -> <UNK>
    for c, i in idx.items():
        if len(c) == 1 and ord(c) < 0x10000:
            lut[ord(c)] = i
    lut[ord("|")] = S.EOS
    lut[1], lut[2] = S.BOS, S.EOS
    parts = []
    for p in paths:
        text = open(os.path.join(ROOT, p), encoding="utf-8").read().strip("\n")
        text = "\x01" + text.replace("\n", "\x02\x01") + "\x02"
        codes = np.frombuffer(text.encode("utf-16-le"), np.uint16)
        parts.append(lut[codes])
        del text, codes
    return np.concatenate(parts)


def flat_ngrams_arr(tok, n):
    """columns (context..., next) of every window that does not cross into the next line"""
    L = len(tok) - n + 1
    ok = np.ones(L, bool)
    for j in range(1, n):
        ok &= tok[j:j + L] != S.BOS
    return [tok[j:j + L][ok].astype(np.int64) for j in range(n)]


def topk_table(ctx, nxt, K):
    """ctx: int64 context ids, nxt: next tokens -> dict ctx -> (tokens, counts, total)"""
    key = ctx * S.V + nxt
    u, cnt = np.unique(key, return_counts=True)
    c, t = u // S.V, u % S.V
    order = np.lexsort((-cnt, c))
    c, t, cnt = c[order], t[order], cnt[order]
    starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    ends = np.r_[starts[1:], len(c)]
    table = {}
    for a, b in zip(starts, ends):
        table[int(c[a])] = (t[a:min(b, a + K)].copy(), cnt[a:min(b, a + K)].copy(), int(cnt[a:b].sum()))
    return table


def main(model_dir="t44k/model_d10m_f", K=4, nseq=300, slots=1 << 22, min4=3):
    q = json.load(open(os.path.join(ROOT, model_dir, "q.json")))
    S.configure(q["config"])
    vocab = json.load(open(os.path.join(ROOT, "data", "mixd", "vocab.json"), encoding="utf-8"))["vocab"]
    idx = {c: i for i, c in enumerate(vocab)}
    idx["□"] = 3
    t0 = time.time()
    tok = encode_flat(TRAIN, idx)
    print(f"training tokens {len(tok):,} ({time.time() - t0:.0f}s)", flush=True)
    p2, p1, nx = flat_ngrams_arr(tok, 3)
    tri = topk_table(p2 * S.V + p1, nx, K)
    del p2, p1, nx
    p3, p2, p1, nx = flat_ngrams_arr(tok, 4)
    h4 = ((p3 * 1000003 + p2) * 1000033 + p1) % slots
    del p3, p2, p1
    four = topk_table(h4, nx, K)
    del tok, nx, h4
    print(f"tri contexts {len(tri):,}, four-gram cells {len(four):,} of {slots:,} ({time.time() - t0:.0f}s)", flush=True)

    # validation: teacher text + dialogues (held out)
    val = []
    for p in ("data/mix64d/val.txt", "data/dialog64d/val.txt"):
        lines = [l.rstrip("\n") for l in open(os.path.join(ROOT, p), encoding="utf-8") if l.strip()]
        random.Random(4).shuffle(lines)
        val += encode_lines([p], idx, limit=None)[:0] or [
            [S.BOS] + [S.EOS if c == "|" else idx.get(c, 3) for c in l][:S.T - 1] + ([] if l.endswith("，") else [S.EOS])
            for l in lines[:nseq // 2]]
    im = S.IntModel(q)
    allowed = np.asarray(q["allowed"])
    rows = []                   # (z over allowed, target position in allowed, ngram logp over allowed)
    pos_in_allowed = {int(t): i for i, t in enumerate(allowed)}
    for s in val:
        cache = im.new_cache()
        for p in range(len(s) - 1):
            _, lg = im.step(s[p], p, cache)
            tgt = s[p + 1]
            if tgt == 3 or tgt not in pos_in_allowed:
                continue
            z = np.asarray(lg, np.float64)[allowed]
            ng = np.full(len(allowed), np.nan)
            hit = None
            if p >= 2:
                c4 = ((s[p - 2] * 1000003 + s[p - 1]) * 1000033 + s[p]) % slots
                if c4 in four and four[c4][2] >= min4:
                    hit = four[c4]
            if hit is None and p >= 1 and (s[p - 1] * S.V + s[p]) in tri:
                hit = tri[s[p - 1] * S.V + s[p]]
            if hit is not None:
                toks, cnts, tot = hit
                ng[:] = 0.0
                for t, c in zip(toks, cnts):
                    if int(t) in pos_in_allowed:
                        ng[pos_in_allowed[int(t)]] = c / tot
            rows.append((z, pos_in_allowed[tgt], ng))
    print(f"validation positions {len(rows):,} ({time.time() - t0:.0f}s)", flush=True)

    def evaluate(rs, s_, w, eps):
        ce = hit = 0.0
        for z, t, ng in rs:
            sc = s_ * z
            if w and not np.isnan(ng[0]):
                sc = sc + w * np.log(ng + eps)
            sc = sc - sc.max()
            lp = sc - np.log(np.exp(sc).sum())
            ce -= lp[t]; hit += int(np.argmax(sc) == t)
        return ce / len(rs), hit / len(rs)

    random.Random(0).shuffle(rows)
    fit, test = rows[:len(rows) // 2], rows[len(rows) // 2:]
    best_s = min(((evaluate(fit[:3000], s_, 0, 1e-3)[0], s_) for s_ in np.geomspace(1e-4, 1e-1, 25)))[1]
    base = evaluate(test, best_s, 0, 1e-3)
    print(f"model only           : CE {base[0]:.4f}  top-1 {base[1]:.4f}   (scale {best_s:.2e})")
    best = None
    for w in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        for eps in (1e-3, 1e-2, 3e-2):
            r = evaluate(fit[:3000], best_s, w, eps)
            if best is None or r[0] < best[0]:
                best = (r[0], w, eps)
    _, w, eps = best
    mix = evaluate(test, best_s, w, eps)
    print(f"model + n-gram top-{K}: CE {mix[0]:.4f}  top-1 {mix[1]:.4f}   (w {w}, eps {eps}) "
          f"-> CE {mix[0] - base[0]:+.4f}, top-1 {100 * (mix[1] - base[1]):+.2f} points")
    cover = sum(1 for _, _, ng in test if not np.isnan(ng[0])) / len(test)
    print(f"positions with an n-gram entry: {cover:.1%}")


if __name__ == "__main__":
    a = sys.argv[1:]
    K = int(a[a.index("--k") + 1]) if "--k" in a else 4
    n = int(a[a.index("--seqs") + 1]) if "--seqs" in a else 300
    d = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in ("--k", "--seqs"))]
    main(d[0] if d else "t44k/model_d10m_f", K, n)
