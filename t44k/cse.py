"""Common-subexpression elimination for ternary matrices (exact integer arithmetic).

Today every nonzero weight is one += / -= . Greedy pair CSE (Paar-style, signed): repeatedly take
the signed pair (x_i, rel*x_j) shared by the most rows (>= MIN_COUNT), compute u = x_i + rel*x_j
once (2 commands) and let those rows use u (one command instead of two). Temporaries may be paired
again. The rows still sum exactly the same terms, so every integer result is unchanged.

Measured on the trained T1M (ctx 64): all matmuls 440,724 -> 261,900 commands (x1.68).

plan(W) -> (temps, rows)
  temps: [(i, rel, j)]  temp k is column n + k = col_i + rel * col_j (columns: inputs, then temps)
  rows : [{column: sign}] for every row of W
Plans are cached in t44k/cse_cache/<sha1 of the matrix>.json (a model takes minutes).
"""
import os, json, hashlib
import numpy as np

MIN_COUNT = 3                      # a pair used by k rows saves k - 2 commands
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cse_cache")


def greedy(M, min_count=MIN_COUNT):
    M = np.asarray(M, np.int8)
    rows = [dict((int(j), int(M[r, j])) for j in np.nonzero(M[r])[0]) for r in range(M.shape[0])]
    n = M.shape[1]
    cap = n + 8192
    same = np.zeros((cap, cap), np.int32)
    opp = np.zeros((cap, cap), np.int32)

    def pairs(row, d):
        """add d to the pair counts of every (i, j), i != j, of this row (vectorised; the column
        indices of one row are distinct, so fancy-index += is exact)"""
        if len(row) < 2:
            return
        ks = np.fromiter(row.keys(), np.int64, len(row))
        ss = np.fromiter(row.values(), np.int64, len(row))
        eq = ss[:, None] == ss[None, :]
        np.fill_diagonal(eq, False)
        ne = ~(ss[:, None] == ss[None, :])
        ix = np.ix_(ks, ks)
        same[ix] += d * eq
        opp[ix] += d * ne
    for row in rows:
        pairs(row, 1)
    where = {}
    for r, row in enumerate(rows):
        for j in row:
            where.setdefault(j, set()).add(r)
    temps = []
    while n < cap:
        s, o = same[:n, :n], opp[:n, :n]
        ms, mo = int(s.max()), int(o.max())
        if max(ms, mo) < min_count:
            break
        if ms >= mo:
            i, j = np.unravel_index(int(np.argmax(s)), (n, n)); rel = 1
        else:
            i, j = np.unravel_index(int(np.argmax(o)), (n, n)); rel = -1
        i, j = int(i), int(j)
        c = n; n += 1
        temps.append((i, rel, j))
        hit = [r for r in where.get(i, set()) & where.get(j, set()) if rows[r][i] * rows[r][j] == rel]
        for r in hit:
            row = rows[r]
            pairs(row, -1)
            sgn = row.pop(i); row.pop(j)
            row[c] = sgn                         # sgn*x_i + sgn*rel*x_j = sgn*u
            pairs(row, 1)
            where[i].discard(r); where[j].discard(r); where.setdefault(c, set()).add(r)
    return temps, rows


def plan(W):
    M = np.asarray(W, np.int8)
    key = hashlib.sha1(M.tobytes() + str(M.shape).encode()).hexdigest()
    path = os.path.join(CACHE, key + ".json")
    if os.path.exists(path):
        d = json.load(open(path))
        return [tuple(t) for t in d["temps"]], [{int(k): v for k, v in r.items()} for r in d["rows"]]
    temps, rows = greedy(M)
    check(M, temps, rows)
    os.makedirs(CACHE, exist_ok=True)
    json.dump({"shape": list(M.shape), "temps": temps, "rows": [{str(k): v for k, v in r.items()} for r in rows]},
              open(path, "w"))
    return temps, rows


def check(M, temps, rows):
    """expand every row back to input coefficients and compare with M"""
    n = M.shape[1]
    exp = list(np.eye(n, dtype=np.int64))    # one identity (a fresh eye per row kept n of them alive: 12 GB at n=1152)
    for i, rel, j in temps:
        exp.append(exp[i] + rel * exp[j])
    for r, row in enumerate(rows):
        v = np.zeros(n, np.int64)
        for c, s in row.items():
            v += s * exp[c]
        assert np.array_equal(v, M[r].astype(np.int64)), f"CSE row {r} does not expand to the matrix row"


def cost(W):
    temps, rows = plan(W)
    return sum(len(r) for r in rows) + 2 * len(temps)
