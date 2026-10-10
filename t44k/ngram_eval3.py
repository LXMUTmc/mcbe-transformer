"""Third n-gram experiment, memory-safe (numpy tables, one order at a time, hard memory cap).

Questions:
  1. leakage: is the gain real, or are validation lines near-copies of training text? Validation
     lines are split into 'novel' (< 50% of their 8-character windows occur in the training text)
     and 'seen'; gains are reported for both, and for three sources (teacher, dialogue, Wikipedia).
  2. size: block memory holds a few million contexts, not 23M: keep only the N most frequent
     contexts per order and see what is left of the gain.
Tables (as ngram_eval2): orders 2 (direct id) and 3, 4 (hashed into 2^24 cells), K = 8 candidates,
backoff longest-first when the context was seen >= 3 times.

  python -m t44k.ngram_eval3 [model_dir]        (memory capped at 10 GB)
"""
import sys, os, json, random, time
import numpy as np
from t44k.memcap import cap_memory
from t44k import spec as S
from t44k.ngram_eval import encode_flat, flat_ngrams_arr, TRAIN, ROOT

SLOTS, K, MINS = 1 << 24, 8, 3
H1, H2 = 1000003, 1000033


def ctx_ids(cols, n):
    if n == 2:
        return cols[0] * S.V + cols[1]
    h = cols[0].copy()
    for c in cols[1:]:
        h = (h * H1 + c) % (1 << 61)
    return (h * H2) % SLOTS


def window_hash(tok, m):
    """running hash of every m-token window that stays inside one line (no <BOS> after its start);
    returns (hash, valid mask, window count) without materialising m int64 columns"""
    L = len(tok) - m + 1
    h = tok[0:L].astype(np.int64)
    ok = np.ones(L, bool)
    for j in range(1, m):
        nxt = tok[j:j + L]
        ok &= nxt != S.BOS
        h *= H1
        h += nxt
        h %= (1 << 61)
    return h, ok, L


def window_keys(tok, n):
    """(context id, next token) of every valid (n+1)-token window; context ids as ctx_ids()"""
    L = len(tok) - n
    if n == 2:
        ok = (tok[1:1 + L] != S.BOS) & (tok[2:2 + L] != S.BOS)
        ctx = tok[0:L].astype(np.int64) * S.V + tok[1:1 + L]
    else:
        ctx, ok, _ = window_hash(tok[:len(tok) - 1], n)
        ctx = ctx[:L]; ok = ok[:L] & (tok[n:n + L] != S.BOS)
        ctx = (ctx * H2) % SLOTS
    nxt = tok[n:n + L].astype(np.int64)
    return ctx[ok], nxt[ok]


def build(tok, n):
    """compact table: sorted context ids, totals, (n_ctx, K) top tokens and counts"""
    ctx, nxt = window_keys(tok, n)
    key = ctx * S.V + nxt
    del ctx, nxt
    u, cnt = np.unique(key, return_counts=True)
    del key
    c, t = u // S.V, (u % S.V).astype(np.int16)
    del u
    order = np.lexsort((-cnt, c))
    c, t, cnt = c[order], t[order], cnt[order].astype(np.int32)
    del order
    starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    ctx = c[starts]
    tot = np.add.reduceat(cnt, starts).astype(np.int64)
    rank = np.arange(len(c)) - np.repeat(starts, np.diff(np.r_[starts, len(c)]))
    keep = rank < K
    row = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(c)]))[keep]
    toks = np.full((len(ctx), K), -1, np.int16); cnts = np.zeros((len(ctx), K), np.int32)
    toks[row, rank[keep]] = t[keep]; cnts[row, rank[keep]] = cnt[keep]
    return {"ctx": ctx, "tot": tot, "toks": toks, "cnts": cnts}


def prune(tab, n_keep):
    if n_keep is None or n_keep >= len(tab["ctx"]):
        return tab
    sel = np.sort(np.argsort(-tab["tot"])[:n_keep])
    return {k: v[sel] for k, v in tab.items()}


def main(model_dir="t44k/model_d10m_f"):
    cap_memory(10)
    q = json.load(open(os.path.join(ROOT, model_dir, "q.json")))
    S.configure(q["config"])
    vocab = json.load(open(os.path.join(ROOT, "data", "mixd", "vocab.json"), encoding="utf-8"))["vocab"]
    idx = {c: i for i, c in enumerate(vocab)}
    idx["□"] = 3
    allowed = np.asarray(q["allowed"])
    pia = np.full(S.V, -1, np.int64); pia[allowed] = np.arange(len(allowed))
    srcs = [("teacher", "data/mix64d/val.txt"), ("dialog", "data/dialog64d/val.txt"), ("wiki", "data/mix64d/aux_val.txt")]
    val, src_of = [], []
    for name, p in srcs:
        lines = [l.rstrip("\n") for l in open(os.path.join(ROOT, p), encoding="utf-8") if l.strip()]
        random.Random(4).shuffle(lines)
        for l in lines[:150]:
            val.append([S.BOS] + [S.EOS if c == "|" else idx.get(c, 3) for c in l][:S.T - 1] + ([] if l.endswith("，") else [S.EOS]))
            src_of.append(name)
    cache = os.path.join(ROOT, "t44k", "ngram_cache3.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        Z, TG, SI, PP = d["Z"], d["TG"], d["SI"], d["PP"]
    else:
        im, Z, TG, SI, PP = S.IntModel(q), [], [], [], []
        for si, s in enumerate(val):
            kv = im.new_cache()
            for p in range(len(s) - 1):
                _, lg = im.step(s[p], p, kv)
                if s[p + 1] == 3 or pia[s[p + 1]] < 0:
                    continue
                Z.append(np.asarray(lg, np.float32)[allowed]); TG.append(pia[s[p + 1]]); SI.append(si); PP.append(p)
        Z, TG, SI, PP = np.stack(Z), np.asarray(TG), np.asarray(SI), np.asarray(PP)
        np.savez(cache, Z=Z, TG=TG, SI=SI, PP=PP)
    print(f"validation positions {len(TG):,}", flush=True)

    t0 = time.time()
    tok = encode_flat(TRAIN, idx)
    # leakage: share of each validation line's 8-character windows that occur in the training text
    h8, ok8, _ = window_hash(tok, 8)
    h8 = np.unique(h8[ok8])
    del ok8
    novel = np.zeros(len(val), bool)
    for si, s in enumerate(val):
        a = np.asarray(s, np.int64)
        if len(a) < 8:
            novel[si] = True; continue
        hh = a[0:len(a) - 7].copy()
        for j in range(1, 8):
            hh = (hh * H1 + a[j:len(a) - 7 + j]) % (1 << 61)
        pos = np.searchsorted(h8, hh)
        seen = (pos < len(h8)) & (h8[np.minimum(pos, len(h8) - 1)] == hh)
        novel[si] = seen.mean() < 0.5
    del h8
    print(f"novel lines: " + ", ".join(f"{n} {novel[[i for i, x in enumerate(src_of) if x == n]].mean():.0%}" for n, _ in srcs)
          + f" ({time.time() - t0:.0f}s)", flush=True)

    tabs = {}
    for n in (2, 3, 4):
        tabs[n] = build(tok, n)
        print(f"order {n}: {len(tabs[n]['ctx']):,} contexts ({time.time() - t0:.0f}s)", flush=True)
    del tok

    def ngram_matrix(tb):
        NG = np.ones(Z.shape, np.float64)                    # 1.0 = no entry (constant term)
        done = np.zeros(len(TG), bool)
        for n in (4, 3, 2):
            if n not in tb:
                continue
            t = tb[n]
            ok = (PP + 1 >= n) & ~done
            rows = np.flatnonzero(ok)
            cols = [np.asarray([val[SI[r]][PP[r] - n + 1 + j] for r in rows], np.int64) for j in range(n)]
            cid = ctx_ids(cols, n)
            pos = np.searchsorted(t["ctx"], cid)
            hit = (pos < len(t["ctx"])) & (t["ctx"][np.minimum(pos, len(t["ctx"]) - 1)] == cid)
            hit &= t["tot"][np.minimum(pos, len(t["ctx"]) - 1)] >= MINS
            for r, p_ in zip(rows[hit], pos[hit]):
                NG[r] = 0.0
                for tt, cc in zip(t["toks"][p_], t["cnts"][p_]):
                    if tt >= 0 and pia[tt] >= 0:
                        NG[r, pia[tt]] = cc / t["tot"][p_]
                done[r] = True
        return NG

    perm = np.random.default_rng(0).permutation(len(TG))
    fit = perm[:len(perm) // 2]

    def ce_acc(rows, s_, w, eps, NG):
        sc = s_ * Z[rows].astype(np.float64)
        if w:
            sc = sc + w * np.log(NG[rows] + eps)
        sc -= sc.max(1, keepdims=True)
        lp = sc - np.log(np.exp(sc).sum(1, keepdims=True))
        return float(-lp[np.arange(len(rows)), TG[rows]].mean()), float((sc.argmax(1) == TG[rows]).mean())

    s_best = min((ce_acc(fit, s_, 0, 0, None)[0], s_) for s_ in np.geomspace(3e-4, 3e-2, 30))[1]
    test = perm[len(perm) // 2:]
    groups = {"all": test}
    for n, _ in srcs:
        groups[n] = test[[src_of[SI[r]] == n for r in test]]
    groups["novel"] = test[novel[SI[test]]]
    groups["seen"] = test[~novel[SI[test]]]
    for label, keep in (("full 23M", None), ("5M", 5_000_000), ("2M", 2_000_000), ("1M", 1_000_000), ("500K", 500_000)):
        tb = {}
        for n in (2, 3, 4):
            share = len(tabs[n]["ctx"]) / sum(len(tabs[m]["ctx"]) for m in (2, 3, 4))
            tb[n] = prune(tabs[n], None if keep is None else int(keep * share))
        NG = ngram_matrix(tb)
        best = min((ce_acc(fit, s_best, w, eps, NG)[0], w, eps) for w in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4) for eps in (1e-4, 1e-3))
        line = f"table {label:9s} ({sum(len(tb[n]['ctx']) for n in tb):>10,} ctx)  w {best[1]:<4} "
        for g, rows in groups.items():
            if len(rows) == 0:
                continue
            b, m = ce_acc(rows, s_best, 0, 0, None), ce_acc(rows, s_best, best[1], best[2], NG)
            line += f"| {g} {100 * (m[1] - b[1]):+.1f}pt {m[0] - b[0]:+.3f} "
        print(line + f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
