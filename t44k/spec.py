"""MCBE-T44K integer specification: the single source of truth for every integer.

The compiler (t44k/compiler.py) emits one scoreboard command per step of this file; the
offline checker runs the emitted command TEXT in vm.VM and compares every tap with IntModel.

Model: decoder-only, 2 layers, d=32, 4 heads x 8, FFN 112 (ReLU), vocab 512, context 16,
Pre-RMSNorm (no affine), ternary linear layers, INT8 token/position embeddings, separate
ternary LM head, greedy argmax. 55,808 nominal parameters (43,520 with the original 320 vocabulary).

Primitives (all map 1:1 to commands): set, add/remove literal, =, +=, -=, *=, /= with a
POSITIVE divisor and a NON-NEGATIVE dividend, '>' (max), range tests, 'a > b' tests.
Rounded division rdiv(x, D) = floor((x + K*D + D//2) / D) - K with K*D <= 2^30, so the
dividend is never negative and the result does not depend on the platform's division.
"""
import json, math, hashlib
import numpy as np

V, T, D, H, DH, F, L = 512, 16, 32, 4, 8, 112, 2    # vocab raised 320 -> 512 (corpus coverage, 2026-09-29)
# model sizes: configure() rebinds the module constants above; every other module reads them as S.X
CONFIGS = {
    "t44k": dict(V=512, T=16, D=32, H=4, DH=8, F=112, L=2),       # 55,808 params, the M6 world
    "t1m":  dict(V=1024, T=32, D=128, H=8, DH=16, F=384, L=6),    # ~1.25M params, staged over ticks
    "t1m64": dict(V=1024, T=64, D=128, H=8, DH=16, F=384, L=6),   # T1M with context 64 (+4K position params)
    "t4m":  dict(V=1024, T=64, D=256, H=8, DH=32, F=768, L=6),    # ~4.47M params; 8 heads keep the softmax cost of T1M
    "t7m":  dict(V=1024, T=64, D=320, H=8, DH=40, F=960, L=6),    # ~6.8M params (scaling study)
    "t10m": dict(V=1024, T=64, D=384, H=8, DH=48, F=1152, L=6),   # ~9.6M params (scaling study)
    "t4m12": dict(V=1024, T=64, D=256, H=8, DH=32, F=768, L=12),  # 12 layers: cross-layer sharing study (train --share 6)
    "mini": dict(V=1024, T=64, D=64, H=4, DH=16, F=192, L=2),     # ~217K params: pipeline smoke tests and quick demos
}
CONFIG = "t44k"
BOS, EOS = 1, 2
S = 64              # residual / normalised activation scale (INT8 embeddings share it)
PS = 4              # pre-scale before squaring in RMSNorm
QS = 4              # pre-scale of q and k before the q.k products
INVR = 4096         # RMSNorm reciprocal fixed point
R = 4096            # default rescale denominator after Wo and W2 (a model may set q["R"])
U = 0.25            # attention score unit (nats per integer step)
EXPW = 256          # exp LUT weight at delta 0
KATT = 1 << 16      # offset multiplier for the variable attention denominator (den <= T*EXPW)
MASKED = -1000000
BAN = 1000000        # decoding: logit malus that rules a token out (punctuation right after punctuation)
RMS_BINS = 48
GUARD = 1 << 28     # |value| guard for every stored integer (int32 limit is 2^31)
N_NORMS = 2 * L + 1  # attn norm and ffn norm per layer, plus the final norm


def configure(cfg):
    """select a model size (a CONFIGS name or a dict); call before building compilers or models"""
    global V, T, D, H, DH, F, L, N_NORMS, CONFIG
    CONFIG = cfg if isinstance(cfg, str) else "custom"
    c = CONFIGS[cfg] if isinstance(cfg, str) else cfg
    assert c["H"] * c["DH"] == c["D"]
    V, T, D, H, DH, F, L = (c[k] for k in ("V", "T", "D", "H", "DH", "F", "L"))
    N_NORMS = 2 * L + 1
    assert T * EXPW * KATT < 2 ** 31, "attention offset overflows"


def label():
    """short model name for in-game messages ([T10M] ...): the size name without the context /
    depth suffix of the variants"""
    return {"t1m64": "T1M", "t4m12": "T4M", "custom": "MCT"}.get(CONFIG, CONFIG.upper())


def koff(Dv):
    """offset multiplier for constant divisor Dv: K*Dv <= 2^30"""
    return (1 << 30) // Dv


def rdiv(x, Dv, K=None):
    K = koff(Dv) if K is None else K
    x = np.asarray(x, dtype=np.int64)
    num = x + K * Dv + Dv // 2
    assert np.all(num >= 0), ("negative dividend", int(num.min()))
    assert np.all(num < 2 ** 31), ("int32 overflow", int(num.max()))
    return num // Dv - K


def exp_lut():
    """[(lo, hi, w)] on delta <= 0, consecutive equal weights merged; deltas below the last row -> 0"""
    rows, d = [], 0
    while True:
        w = int(round(EXPW * math.exp(d * U)))
        if w == 0:
            break
        rows.append((d, w)); d -= 1
    merged = []
    for d, w in rows:
        if merged and merged[-1][2] == w:
            merged[-1][0] = d
        else:
            merged.append([d, d, w])
    return [(lo, hi, w) for lo, hi, w in merged]


def rms_lut(mlo, mhi, bins=RMS_BINS):
    """geometric bins on ms = mean((x/PS)^2); inv such that x*inv/INVR ~= S*x/rms(x)"""
    mlo = max(1, int(mlo)); mhi = max(mlo + bins, int(mhi))
    edges = np.unique(np.round(np.geomspace(mlo, mhi, bins + 1)).astype(np.int64))
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = int(edges[i]), int(edges[i + 1]) - 1
        if hi < lo:
            continue
        mid = math.sqrt(max(lo, 0.5) * (hi + 1))
        rows.append([lo, hi, int(round(S * INVR / (PS * math.sqrt(mid))))])
    rows[0][0] = 0
    rows[-1][1] = None
    return [tuple(r) for r in rows]


def lut_lookup(lut, x):
    for lo, hi, v in lut:
        if x >= lo and (hi is None or x <= hi):
            return v
    return 0


def cst(q, name, l):
    """model constant for layer l: a scalar or a per-layer list"""
    v = q[name]
    return v[l] if isinstance(v, (list, tuple)) else v


def tern(Wm, x):
    return (np.asarray(Wm, np.int64) @ np.asarray(x, np.int64)).astype(np.int64)


def check_range(name, arr):
    a = np.asarray(arr, np.int64)
    assert np.all(np.abs(a) < GUARD), f"range guard: {name} max |.| = {int(np.abs(a).max())}"


# ----------------------------------------------------------------------------- models
def random_model(seed=0, zero_frac=0.6):
    """engine-validation model: random INT8 embeddings, random ternary matrices, constants chosen
    so activations stay in a realistic range. Used for M0-M4 before a trained model exists."""
    rng = np.random.default_rng(seed)

    def ternary(o, i):
        m = rng.choice([-1, 1], size=(o, i))
        m[rng.random((o, i)) < zero_frac] = 0
        return m.astype(int).tolist()

    q = {"config": dict(V=V, T=T, D=D, H=H, DH=DH, F=F, L=L), "seed": seed, "kind": "random"}
    q["E"] = np.clip(np.round(rng.normal(0, 40, (V, D))), -127, 127).astype(int).tolist()
    q["P"] = np.clip(np.round(rng.normal(0, 20, (T, D))), -127, 127).astype(int).tolist()
    for l in range(L):
        for m in ("Wq", "Wk", "Wv", "Wo"):
            q[f"{m}{l}"] = ternary(D, D)
        q[f"W1{l}"] = ternary(F, D)
        q[f"W2{l}"] = ternary(D, F)
    q["Wh"] = ternary(V, D)
    # constants tuned on T44K (D=32, DH=8, F=112); magnitudes grow with the number of nonzero terms
    rD, rF = D / 32, F / 112
    q["ATTDIV"] = int(3000 * rD * math.sqrt(DH / 8))   # q.k (pre-scaled) -> attention units
    q["MO"] = max(1, int(300 / rD))                     # (Wo a) * MO / R -> residual scale
    q["M2"] = max(1, int(240 / math.sqrt(rD * rF)))     # (W2 f) * M2 / R -> residual scale
    q["exp_lut"] = [list(r) for r in exp_lut()]
    q["allowed"] = [EOS] + list(range(4, V))
    q["rms_luts"] = None
    return q


class IntModel:
    def __init__(self, q):
        self.q = q
        # weight matrices as int64 arrays once (converting the JSON lists on every step dominated
        # the run time at T1M size); same integers
        self.W = {k: np.asarray(v, np.int64) for k, v in q.items()
                  if k in ("E", "P", "Wh") or (k[:2] in ("Wq", "Wk", "Wv", "Wo", "W1", "W2") and k[2:].isdigit())}
        self.explut = [tuple(r) for r in q["exp_lut"]]
        self.luts = [[tuple(r) for r in l] for l in q["rms_luts"]] if q.get("rms_luts") else None
        self.ms_log = [[] for _ in range(N_NORMS)]

    def rms(self, x, which):
        xs = rdiv(x, PS)
        ss = int(np.sum(xs * xs))
        ms = int(rdiv(ss, D))
        self.ms_log[which].append(ms)
        if self.luts is None:                         # calibration pass
            inv = int(round(S * INVR / (PS * math.sqrt(max(ms, 1)))))
        else:
            inv = lut_lookup(self.luts[which], ms)
        check_range("x*inv", np.asarray(x, np.int64) * inv)
        return rdiv(np.asarray(x, np.int64) * inv, INVR), dict(ss=ss, ms=ms, inv=inv)

    def new_cache(self):
        return {"K": np.zeros((L, T, D), np.int64), "V": np.zeros((L, T, D), np.int64)}

    def step(self, tok, pos, cache, trace=None):
        q = self.q
        tr = {}
        W = self.W
        x = W["E"][tok] + W["P"][pos]
        tr["xe"] = x.copy()
        for l in range(L):
            n, st = self.rms(x, 2 * l); tr[f"na{l}"] = n; tr[f"rms_a{l}"] = st
            qv = tern(W[f"Wq{l}"], n); kv = tern(W[f"Wk{l}"], n); vv = tern(W[f"Wv{l}"], n)
            qs, ks = rdiv(qv, QS), rdiv(kv, QS)
            cache["K"][l, pos] = ks; cache["V"][l, pos] = vv
            a = np.zeros(D, np.int64)
            S_, W_ = [], []
            n_live = pos + 1
            for h in range(H):                        # vectorised over slots; same integers as a scalar loop
                sl = slice(DH * h, DH * h + DH)
                dots = cache["K"][l, :n_live, sl] @ qs[sl]
                check_range("qk", dots)
                s = [int(v) for v in rdiv(dots, cst(q, "ATTDIV", l))] + [MASKED] * (T - n_live)
                smax = max(s)
                w = [lut_lookup(self.explut, sj - smax) for sj in s]
                den = sum(w)
                num = np.asarray(w, np.int64) @ cache["V"][l, :, sl]
                check_range("attn num", num)
                a[sl] = rdiv(num, den, KATT)
                S_.append(s); W_.append(w)
            o = rdiv(tern(W[f"Wo{l}"], a) * cst(q, "MO", l), q.get("R", R))
            x = x + o
            tr.update({f"q{l}": qv, f"k{l}": kv, f"v{l}": vv, f"qs{l}": qs, f"ks{l}": ks, f"s{l}": S_, f"w{l}": W_,
                       f"a{l}": a.copy(), f"o{l}": o, f"xa{l}": x.copy()})
            n2, st = self.rms(x, 2 * l + 1); tr[f"nb{l}"] = n2; tr[f"rms_b{l}"] = st
            z = tern(W[f"W1{l}"], n2)
            f = np.maximum(z, 0)
            g = rdiv(tern(W[f"W2{l}"], f) * cst(q, "M2", l), q.get("R", R))
            x = x + g
            tr.update({f"f{l}": f, f"g{l}": g, f"xf{l}": x.copy()})
            for nm in ("na", "nb", "q", "k", "v", "a", "f", "xf"):
                check_range(nm, tr[f"{nm}{l}"])
        h3, st = self.rms(x, 2 * L); tr["nf"] = h3; tr["rms_f"] = st
        logits = tern(W["Wh"], h3)
        allowed = q["allowed"]
        best, bl = allowed[0], logits[allowed[0]]
        for t in allowed[1:]:
            if logits[t] > bl:
                best, bl = t, logits[t]
        tr.update(best=best, bestLog=int(bl))
        if trace is not None:
            trace.append(tr)
        return best, logits

    def teacher_forced(self, seq):
        """greedy prediction at every position of seq (teacher forcing)"""
        cache = self.new_cache()
        return [self.step(t, pos, cache)[0] for pos, t in enumerate(seq[:T])]

    def generate(self, prompt_ids, max_new=None, trace=None, penalty=None, exempt=None, turn_reset=None):
        """prompt_ids starts with BOS. Prefill then greedy until EOS / context full / max_new.
        penalty > 0: presence penalty -- every allowed token that was already fed into the model
        (prompt or generated) loses `penalty` integer logit units; tokens in `exempt` (punctuation)
        are never penalised. Ties still go to the lowest id (strict '>').
        turn_reset (dialogue): an <EOS> fed into the model starts a new turn and clears the
        'seen' set, so the penalty only acts within the current turn (a reply may quote the
        other speaker's words, but does not loop on its own)."""
        max_new = T if max_new is None else max_new          # T of the configured size, not the import-time one
        dec = self.q.get("decode") or {}
        penalty = dec.get("penalty", 0) if penalty is None else penalty
        exempt = set(dec.get("exempt", ())) if exempt is None else exempt
        turn_reset = dec.get("turn_reset", False) if turn_reset is None else turn_reset
        cache = self.new_cache()
        out, cur = [], prompt_ids[0]
        seen = set()
        for pos in range(T):
            if turn_reset and cur == EOS:
                seen = set()
            seen.add(cur)
            best, logits = self.step(cur, pos, cache, trace)
            if penalty:
                al = self.q["allowed"]
                # presence penalty (punctuation exempt); right after a punctuation mark no punctuation
                # mark may follow (BAN): no '，，' or '，。'
                after_p = cur in exempt
                score = [int(logits[t]) - (penalty if (t in seen and t not in exempt) else 0)
                         - (BAN if (after_p and t in exempt and t != EOS) else 0) for t in al]
                bi = 0
                for i in range(1, len(al)):
                    if score[i] > score[bi]:
                        bi = i
                best = al[bi]
                if trace is not None:                 # the game snapshots the penalised winner
                    trace[-1]["best"], trace[-1]["bestLog"] = best, score[bi]
            if pos + 1 < len(prompt_ids):
                cur = prompt_ids[pos + 1]
            else:
                out.append(best)
                if best == EOS or len(out) >= max_new:
                    break
                cur = best
        return out


def calibrate(q, prompts):
    im = IntModel(q)
    for p in prompts:
        im.generate(p)
    luts = []
    for w in range(N_NORMS):
        v = np.array(im.ms_log[w])
        lo, hi = np.percentile(v, 0.5), np.percentile(v, 99.5)
        luts.append([list(r) for r in rms_lut(lo * 0.7, hi * 1.4)])
    q["rms_luts"] = luts
    return q


def random_prompts(n, seed=1, max_len=4):
    rng = np.random.default_rng(seed)
    return [[BOS] + rng.integers(4, V, size=int(rng.integers(1, max_len + 1))).tolist() for _ in range(n)]


def qhash(q):
    return hashlib.sha256(json.dumps(q, sort_keys=True).encode()).hexdigest()[:16]


def nominal_params():
    return V * D + T * D + L * (4 * D * D + 2 * D * F) + V * D
