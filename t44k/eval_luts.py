"""Offline quality check of the integer approximations (no game needed).

Teacher-forced next-character accuracy of the INTEGER model on validation text, with different
RMSNorm inverse-sqrt tables:
  A  the model's current LUTs (spec.RMS_BINS bins, calibrated after training)
  B  exact inverse sqrt (reference; what the LUT approximates)
  C  denser LUTs (N bins, recalibrated on the same data)
plus the share of generations that end with EOS within the context ('normal end' rate).

  python -m t44k.eval_luts CONFIG [n_val] [bins_C]      (CONFIG with a trained model: t1m)
"""
import sys, json, copy, random
import numpy as np
from t44k import spec as S
from t44k.check_staged import TRAINED

UNK = 3


def load(config):
    S.configure(config)
    qpath, vpath = TRAINED[config]
    q = json.load(open(qpath))
    vocab = json.load(open(vpath, encoding="utf-8"))["vocab"]
    idx = {c: i for i, c in enumerate(vocab)}
    idx["□"] = UNK
    val = [l.strip() for l in open(vpath.replace("vocab.json", "val.txt"), encoding="utf-8") if l.strip()]
    seqs = [[S.BOS] + [idx[c] for c in s][:S.T - 1] + ([] if s.endswith("，") else [S.EOS]) for s in val]
    return q, vocab, seqs


def accuracy(q, seqs):
    im = S.IntModel(q)
    hit = tot = 0
    for s in seqs:
        pred = im.teacher_forced(s[:-1][:S.T])
        for p, t in zip(pred, s[1:][:S.T]):
            if t != UNK:
                hit += p == t; tot += 1
    return hit / tot


def eos_rate(q, seqs, n=100):
    im = S.IntModel(q)
    ends = 0
    for s in seqs[:n]:
        out = im.generate(s[:3])
        ends += bool(out) and out[-1] == S.EOS
    return ends / n


def recalibrated(q, seqs, bins):
    q2 = copy.deepcopy(q)
    q2["rms_luts"] = None
    im = S.IntModel(q2)
    for s in seqs:
        im.teacher_forced(s[:-1][:S.T])
    luts = []
    for w in range(S.N_NORMS):
        v = np.array(im.ms_log[w])
        lo, hi = np.percentile(v, 0.5), np.percentile(v, 99.5)
        luts.append([list(r) for r in S.rms_lut(lo * 0.7, hi * 1.4, bins=bins)])
    q2["rms_luts"] = luts
    return q2


def main(config="t1m", n_val=400, bins_c=96):
    q, vocab, seqs = load(config)
    random.Random(0).shuffle(seqs)
    test, cal = seqs[:n_val], seqs[n_val:n_val + 600]
    qa = q
    qb = copy.deepcopy(q); qb["rms_luts"] = None                    # exact inverse sqrt
    qc = recalibrated(q, cal, bins_c)
    rows = []
    for name, qq in (("A current LUT (%d bins)" % len(q["rms_luts"][0]), qa), ("B exact 1/sqrt", qb), ("C %d bins" % bins_c, qc)):
        rows.append((name, accuracy(qq, test), eos_rate(qq, test)))
        print(f"{name:28s} next-char acc {rows[-1][1]:.4f}   normal-end rate {rows[-1][2]:.2f}", flush=True)
    a, b, c = (r[1] for r in rows)
    print(f"reference gain B-A = {b - a:+.4f};  recovered by C = {(c - a) / (b - a) if b != a else float('nan'):.0%}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "t1m", int(a[1]) if len(a) > 1 else 400, int(a[2]) if len(a) > 2 else 96)
