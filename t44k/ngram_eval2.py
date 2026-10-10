"""Second n-gram experiment (see ngram_eval.py): the model's integer logits are computed ONCE and
cached (t44k/ngram_cache.npz), then several block-memory table designs are compared:
  orders  : context lengths used, longest first (backoff when the context was seen < min times)
  K       : candidates kept per context
  slots   : hashed table size for contexts longer than 2 (2 = direct address V*V)
Combination as before: score = s*z + w*log(p + eps) over the allowed tokens; s, w, eps fitted on
half of the positions, reported on the other half.

  python -m t44k.ngram_eval2 [model_dir]
"""
import sys, os, json, random, time
import numpy as np
from t44k import spec as S
from t44k.ngram_eval import encode_flat, flat_ngrams_arr, topk_table, TRAIN, ROOT

H1, H2 = 1000003, 1000033


def ctx_hash(cols, n, slots):
    """cols: context columns (oldest first) -> hashed id (n = 2 uses the direct id)"""
    if n == 2:
        return cols[0] * S.V + cols[1]
    h = cols[0].copy()
    for c in cols[1:]:
        h = (h * H1 + c) % (1 << 61)
    return (h * H2) % slots


def build_tables(tok, orders, K, slots):
    tabs = {}
    for n in orders:
        cols = flat_ngrams_arr(tok, n + 1)
        tabs[n] = topk_table(ctx_hash(cols[:-1], n, slots), cols[-1], K)
        del cols
    return tabs


def lookup(tabs, seq, p, orders, slots, mins):
    for n in sorted(orders, reverse=True):
        if p + 1 < n:
            continue
        cols = [np.asarray([seq[p - n + 1 + j]], np.int64) for j in range(n)]
        key = int(ctx_hash(cols, n, slots)[0])
        hit = tabs[n].get(key)
        if hit is not None and hit[2] >= mins:
            return hit
    return None


def main(model_dir="t44k/model_d10m_f", nseq=300):
    q = json.load(open(os.path.join(ROOT, model_dir, "q.json")))
    S.configure(q["config"])
    vocab = json.load(open(os.path.join(ROOT, "data", "mixd", "vocab.json"), encoding="utf-8"))["vocab"]
    idx = {c: i for i, c in enumerate(vocab)}
    idx["□"] = 3
    allowed = np.asarray(q["allowed"])
    pia = {int(t): i for i, t in enumerate(allowed)}
    cache = os.path.join(ROOT, "t44k", "ngram_cache.npz")
    val = []
    for p in ("data/mix64d/val.txt", "data/dialog64d/val.txt"):
        lines = [l.rstrip("\n") for l in open(os.path.join(ROOT, p), encoding="utf-8") if l.strip()]
        random.Random(4).shuffle(lines)
        val += [[S.BOS] + [S.EOS if c == "|" else idx.get(c, 3) for c in l][:S.T - 1] + ([] if l.endswith("，") else [S.EOS])
                for l in lines[:nseq // 2]]
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        Z, TG, POS = d["Z"], d["TG"], list(d["POS"])
    else:
        im, Z, TG, POS = S.IntModel(q), [], [], []
        for si, s in enumerate(val):
            kv = im.new_cache()
            for p in range(len(s) - 1):
                _, lg = im.step(s[p], p, kv)
                if s[p + 1] == 3 or s[p + 1] not in pia:
                    continue
                Z.append(np.asarray(lg, np.float32)[allowed]); TG.append(pia[s[p + 1]]); POS.append((si, p))
        Z, TG = np.stack(Z), np.asarray(TG)
        np.savez(cache, Z=Z, TG=TG, POS=np.asarray(POS, dtype=object))
    print(f"validation positions {len(TG):,}", flush=True)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(TG))
    fit, test = perm[:len(perm) // 2], perm[len(perm) // 2:]

    def ce_acc(rows, s_, w, eps, NG):
        sc = s_ * Z[rows].astype(np.float64)
        if w:
            sc = sc + w * np.log(NG[rows] + eps)
        sc -= sc.max(1, keepdims=True)
        lp = sc - np.log(np.exp(sc).sum(1, keepdims=True))
        return float(-lp[np.arange(len(rows)), TG[rows]].mean()), float((sc.argmax(1) == TG[rows]).mean())

    s_best = min((ce_acc(fit, s_, 0, 0, None)[0], s_) for s_ in np.geomspace(3e-4, 3e-2, 30))[1]
    base = ce_acc(test, s_best, 0, 0, None)
    print(f"model only: CE {base[0]:.4f} top-1 {base[1]:.4f}")
    t0 = time.time()
    tok = encode_flat(TRAIN, idx)
    configs = [((2,), 4, 1 << 22, 1), ((2, 3), 4, 1 << 22, 3), ((2, 3), 8, 1 << 24, 3), ((2, 3, 4), 8, 1 << 24, 3),
               ((2, 3, 4, 5), 8, 1 << 24, 3), ((2, 4, 6), 8, 1 << 24, 2)]
    built = {}
    for orders, K, slots, mins in configs:
        need = tuple((n, K, slots if n > 2 else 0) for n in orders)
        tabs = {}
        for n, k, sl in need:
            if (n, k, sl) not in built:
                cols = flat_ngrams_arr(tok, n + 1)
                built[(n, k, sl)] = topk_table(ctx_hash(cols[:-1], n, sl or 1), cols[-1], k)
                del cols
            tabs[n] = built[(n, k, sl)]
        NG = np.full(Z.shape, 0.0)
        cov = 0
        for r, (si, p) in enumerate(POS):
            hit = lookup(tabs, val[si], p, orders, slots, mins)
            if hit is None:
                NG[r] = np.nan
                continue
            cov += 1
            toks, cnts, tot = hit
            for t, c in zip(toks, cnts):
                if int(t) in pia:
                    NG[r, pia[int(t)]] = c / tot
        has = ~np.isnan(NG[:, 0])
        NG[~has] = 1.0                     # no entry: a flat (constant) term, i.e. no change
        best = None
        for w in (0.02, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5):
            for eps in (1e-4, 1e-3, 1e-2):
                r = ce_acc(fit, s_best, w, eps, NG)
                if best is None or r[0] < best[0]:
                    best = (r[0], w, eps)
        mix = ce_acc(test, s_best, best[1], best[2], NG)
        cells = sum(len(built[x]) for x in need)
        print(f"orders {orders} K {K} slots {slots:>9,} min {mins}: CE {mix[0]:.4f} ({mix[0] - base[0]:+.4f})  "
              f"top-1 {mix[1]:.4f} ({100 * (mix[1] - base[1]):+.2f})  w {best[1]} eps {best[2]}  coverage {cov / len(POS):.1%}  "
              f"table contexts {cells:,}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
