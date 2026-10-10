"""How many commands does common-subexpression elimination (t44k/cse.py) save on the real matrices?

  python -m t44k.analyze_cse [q.json] [config]
Measured on the trained T1M (ctx 64, 2026-10-02): all matmuls 440,724 -> 261,900 commands (x1.68);
random 60%-zero ternary at width 256: x1.7 (256x256) to x1.9 (1024x256).
"""
import sys, json, time
import numpy as np
from t44k import spec as S
from t44k.cse import greedy


def greedy_cse(M):
    temps, rows = greedy(M)
    return sum(len(r) for r in rows) + 2 * len(temps), len(temps)


def main(qpath="t44k/model_t1m64/q.json", config="t1m64"):
    S.configure(config)
    q = json.load(open(qpath))
    mats = [(f"{k}{l}", np.array(q[f"{k}{l}"], np.int8)) for l in range(S.L) for k in ("Wq", "Wk", "Wv", "Wo", "W1", "W2")]
    mats.append(("Wh", np.array(q["Wh"], np.int8)[q["allowed"]]))
    tot_s = tot_c = 0
    for name, M in mats:
        t0 = time.time()
        s = int(np.count_nonzero(M))
        c, t = greedy_cse(M)
        tot_s += s; tot_c += c
        print(f"{name:5s} {M.shape[0]:5d}x{M.shape[1]:<4d} scalar {s:7d}  CSE {c:7d} ({t} temps)  x{s / c:.2f}  [{time.time() - t0:.0f}s]", flush=True)
    print(f"ALL matmul commands: scalar {tot_s:,}  CSE {tot_c:,}  x{tot_s / tot_c:.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
