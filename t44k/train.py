"""Quantization-aware training of T44K that mirrors the integer spec (t44k/spec.py).

The forward pass works directly in the spec's integer units (float tensors that hold integers;
every rounding is round-with-straight-through-gradient):
  x = E[tok] + P[pos]                (INT8 embeddings, residual scale S)
  n = round(S * x / rms(x))          (RMSNorm; the game uses a calibrated 1/sqrt LUT)
  q,k,v = T n   (T ternary, 60% zeros, from latent weights by magnitude threshold)
  s = round((round(q/QS) . round(k/QS)) / ATTDIV)   attention units of U nats, causal
  a = round(sum softmax(U*s) v)
  x += round((To a) * MO / R) ;  FFN: f = relu(T1 n2) ; x += round((T2 f) * M2 / R)
  logits = Th h          (argmax only in the game; a learned temperature for the loss here)
ATTDIV, MO, M2 are learned per layer (log-parameterised) and rounded to integers on export,
with R = 65536 so the rounding costs < 0.1%.

  python -m t44k.train [--steps N] [--device cuda]
Outputs t44k/model/: q.json (integer model for the compiler), train_log.json
"""
import argparse, json, math, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from t44k import spec as S

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
R_TRAIN = 65536
UNK, UNK_CH = 3, "□"
TURN = "|"                                 # dialogue turn separator in text files -> <EOS>


def ste_round(x):
    return x + (torch.round(x) - x).detach()


def ternary(w, zero_frac):
    a = w.detach().abs().flatten()
    k = max(1, int(round(zero_frac * a.numel())))
    thr = a.kthvalue(k).values
    t = torch.sign(w.detach()) * (w.detach().abs() > thr).float()
    return w + (t - w).detach()                      # forward: ternary; backward: identity


class T44K(nn.Module):
    def __init__(self, zero_frac=0.6, share=None):
        """share=K: only K distinct weight sets for the L layers (layer l uses set l*K//L, e.g. L=12,
        K=6 -> AABBCC...); per-layer scales (ATTDIV, MO, M2) and norms stay per layer"""
        super().__init__()
        self.zf = zero_frac
        self.K = share or S.L
        self.E = nn.Parameter(torch.randn(S.V, S.D) * 40)
        self.P = nn.Parameter(torch.randn(S.T, S.D) * 10)
        lin = lambda o, i: nn.Parameter(torch.randn(o, i))
        self.Wq = nn.ParameterList([lin(S.D, S.D) for _ in range(self.K)])
        self.Wk = nn.ParameterList([lin(S.D, S.D) for _ in range(self.K)])
        self.Wv = nn.ParameterList([lin(S.D, S.D) for _ in range(self.K)])
        self.Wo = nn.ParameterList([lin(S.D, S.D) for _ in range(self.K)])
        self.W1 = nn.ParameterList([lin(S.F, S.D) for _ in range(self.K)])
        self.W2 = nn.ParameterList([lin(S.D, S.F) for _ in range(self.K)])
        self.Wh = lin(S.V, S.D)
        # initial constants tuned on T44K (D=32, DH=8, F=112), scaled by the number of summed terms
        rD, rF = S.D / 32, S.F / 112
        self.log_att = nn.Parameter(torch.full((S.L,), math.log(3000.0 * rD * math.sqrt(S.DH / 8))))
        self.log_mo = nn.Parameter(torch.full((S.L,), math.log(0.04 * R_TRAIN / rD)))
        self.log_m2 = nn.Parameter(torch.full((S.L,), math.log(0.03 * R_TRAIN / math.sqrt(rD * rF))))
        self.log_temp = nn.Parameter(torch.tensor(math.log(1 / (400.0 * math.sqrt(rD)))))
        mask = torch.tril(torch.ones(S.T, S.T, dtype=torch.bool))
        self.register_buffer("causal", mask)

    def ws(self, l):
        return l * self.K // S.L

    def rms(self, x):
        return ste_round(S.S * x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1.0))

    def T(self, w):
        return ternary(w, self.zf)

    def forward(self, tok):
        B, Tn = tok.shape
        E = ste_round(self.E).clamp(-127, 127)
        P = ste_round(self.P).clamp(-127, 127)
        x = E[tok] + P[:Tn].unsqueeze(0)
        for l in range(S.L):
            n = self.rms(x)
            w = self.ws(l)
            q = n @ self.T(self.Wq[w]).t(); k = n @ self.T(self.Wk[w]).t(); v = n @ self.T(self.Wv[w]).t()
            qs, ks = ste_round(q / S.QS), ste_round(k / S.QS)
            qh = qs.view(B, Tn, S.H, S.DH).transpose(1, 2)
            kh = ks.view(B, Tn, S.H, S.DH).transpose(1, 2)
            vh = v.view(B, Tn, S.H, S.DH).transpose(1, 2)
            s = ste_round((qh @ kh.transpose(-1, -2)) / self.log_att[l].exp())
            s = s.masked_fill(~self.causal[:Tn, :Tn], -1e9)
            p = torch.softmax(S.U * s, dim=-1)
            a = ste_round(p @ vh).transpose(1, 2).reshape(B, Tn, S.D)
            o = ste_round((a @ self.T(self.Wo[w]).t()) * self.log_mo[l].exp() / R_TRAIN)
            x = x + o
            n2 = self.rms(x)
            f = F.relu(n2 @ self.T(self.W1[w]).t())
            g = ste_round((f @ self.T(self.W2[w]).t()) * self.log_m2[l].exp() / R_TRAIN)
            x = x + g
        h = self.rms(x)
        return (h @ self.T(self.Wh).t()) * self.log_temp.exp()

    def export(self, allowed):
        tern = lambda w: ternary(w, self.zf).detach().round().long().cpu().tolist()
        q = {"config": dict(V=S.V, T=S.T, D=S.D, H=S.H, DH=S.DH, F=S.F, L=S.L), "kind": "trained",
             "E": torch.round(self.E).clamp(-127, 127).long().cpu().tolist(),
             "P": torch.round(self.P).clamp(-127, 127).long().cpu().tolist(), "R": R_TRAIN,
             "ATTDIV": [int(round(v)) for v in self.log_att.exp().tolist()],
             "MO": [int(round(v)) for v in self.log_mo.exp().tolist()],
             "M2": [int(round(v)) for v in self.log_m2.exp().tolist()]}
        for l in range(S.L):
            w = self.ws(l)
            q[f"Wq{l}"], q[f"Wk{l}"], q[f"Wv{l}"], q[f"Wo{l}"] = (tern(m[w]) for m in (self.Wq, self.Wk, self.Wv, self.Wo))
            q[f"W1{l}"], q[f"W2{l}"] = tern(self.W1[w]), tern(self.W2[w])
        q["share"] = self.K
        q["Wh"] = tern(self.Wh)
        q["exp_lut"] = [list(r) for r in S.exp_lut()]
        q["allowed"] = allowed
        q["rms_luts"] = None
        return q


def encode(path, idx):
    """compact store: int16 array (N, T+1), -1 = padding (1.4M Wikipedia chunks ~ 90 MB instead
    of several GB of Python lists)"""
    lines = [l.strip() for l in open(path, encoding="utf-8")]
    lines = [s for s in lines if s]
    arr = np.full((len(lines), S.T + 1), -1, dtype=np.int16)
    for i, s in enumerate(lines):
        # a chunk cut at a comma is not the end of a sentence: no EOS target after it
        tail = [] if s.endswith("，") else [S.EOS]
        seq = ([S.BOS] + [idx[c] for c in s][:S.T - 1] + tail)[:S.T + 1]
        arr[i, :len(seq)] = seq
    return arr


def load_data(data_dir=None, aux=False):
    """default: the T44K corpus (data/vocab_t44k.json, train.txt, val.txt);
    data_dir: a directory with vocab.json, train.txt, val.txt (e.g. data/wiki from prep_wiki.py,
    data/mix from prep_mix.py). aux=True also returns aux_train.txt / aux_val.txt (None if absent)"""
    vpath = os.path.join(data_dir, "vocab.json") if data_dir else os.path.join(DATA, "vocab_t44k.json")
    base = data_dir or DATA
    vocab = json.load(open(vpath, encoding="utf-8"))["vocab"]
    assert len(vocab) == S.V, f"vocabulary {len(vocab)} != model V {S.V}"
    idx = {c: i for i, c in enumerate(vocab)}
    idx[UNK_CH] = UNK                      # data/prep_wiki.py writes out-of-vocabulary characters as '□'
    idx[TURN] = S.EOS                      # data/prep_dialog.py separates dialogue turns with '|'
    out = [vocab, encode(os.path.join(base, "train.txt"), idx), encode(os.path.join(base, "val.txt"), idx)]
    if aux:
        p = os.path.join(base, "aux_train.txt")
        out += [encode(p, idx), encode(os.path.join(base, "aux_val.txt"), idx)] if os.path.exists(p) else [None, None]
    return tuple(out)


def seq(arr, i):
    """one sequence as a Python list (BOS ... [EOS])"""
    r = arr[i]
    return [int(v) for v in r[r >= 0]]


def batchify(arr, device, with_mask=False):
    """arr: int16 (B, T+1) rows from the store (numpy, or a torch tensor already on the device);
    with_mask: also the (B, T) mask of real (non-padding) next positions, for distillation"""
    a = arr.long() if torch.is_tensor(arr) else torch.from_numpy(arr.astype(np.int64))
    x, y = a[:, :S.T].clone(), a[:, 1:S.T + 1].clone()
    m = y >= 0
    x[x < 0] = 0
    y[y < 0] = -100
    y[y == UNK] = -100                     # <UNK> may appear in the context, never as a target
    return (x.to(device), y.to(device), m.to(device)) if with_mask else (x.to(device), y.to(device))


def mixed_batch(main, aux, frac, batch, device, extra=None, extra_frac=0.0):
    """rows from the main store, a fraction `frac` from the auxiliary store and `extra_frac` from the
    extra store (all on device)"""
    na = int(round(batch * frac)) if aux is not None else 0
    ne = int(round(batch * extra_frac)) if extra is not None else 0
    parts = [main[torch.randint(0, len(main), (batch - na - ne,), device=device)]]
    if na:
        parts.append(aux[torch.randint(0, len(aux), (na,), device=device)])
    if ne:
        parts.append(extra[torch.randint(0, len(extra), (ne,), device=device)])
    return torch.cat(parts) if len(parts) > 1 else parts[0]


def mixed_batch_n(stores, fracs, batch, device):
    """rows from several stores (on device), store k contributing round(batch * fracs[k]) rows (the
    first store takes the rounding remainder)"""
    n = [int(round(batch * f)) for f in fracs]
    n[0] = batch - sum(n[1:])
    parts = [st[torch.randint(0, len(st), (k,), device=device)] for st, k in zip(stores, n) if k > 0]
    return torch.cat(parts) if len(parts) > 1 else parts[0]


def load_mix(specs, vocab_path=None):
    """--mix DIR=FRAC ...: every DIR has train.txt / val.txt in one vocabulary (vocab_path, or the first
    DIR's vocab.json). Returns vocab and [(name, frac, train, val, is_dialogue)]; a source is dialogue
    when its text separates turns with '|'."""
    pairs = []
    for sp in specs:
        d, f = sp.rsplit("=", 1)
        pairs.append((d, float(f)))
    tot = sum(f for _, f in pairs)
    vocab = json.load(open(vocab_path or os.path.join(pairs[0][0], "vocab.json"), encoding="utf-8"))["vocab"]
    assert len(vocab) == S.V, f"vocabulary {len(vocab)} != model V {S.V}"
    idx = {c: i for i, c in enumerate(vocab)}
    idx[UNK_CH] = UNK
    idx[TURN] = S.EOS
    out = []
    for d, f in pairs:
        head = "".join(l for _, l in zip(range(2000), open(os.path.join(d, "train.txt"), encoding="utf-8")))
        out.append((os.path.basename(os.path.normpath(d)), f / tot, encode(os.path.join(d, "train.txt"), idx),
                    encode(os.path.join(d, "val.txt"), idx), TURN in head))
    return vocab, out


def load_extra(data_dir, vocab):
    """an extra corpus (e.g. data/dialog64d from prep_dialog.py) encoded with the main vocabulary"""
    idx = {c: i for i, c in enumerate(vocab)}
    idx[UNK_CH] = UNK
    idx[TURN] = S.EOS
    return encode(os.path.join(data_dir, "train.txt"), idx), encode(os.path.join(data_dir, "val.txt"), idx)


def kd_loss(student_logits, teacher_logits, mask, temp=1.0):
    """KL(teacher || student) at temperature temp over the allowed tokens, mean over real positions
    (both logit tensors already have the not-allowed tokens masked to -1e9)"""
    pt = F.softmax(teacher_logits / temp, -1)
    ls = F.log_softmax(student_logits / temp, -1)
    kl = (pt * (torch.log(pt.clamp_min(1e-9)) - ls)).sum(-1)
    return (kl * mask).sum() / mask.sum().clamp_min(1) * temp * temp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--zero-frac", type=float, default=0.6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="t44k", help="model size (spec.CONFIGS)")
    ap.add_argument("--data", default=None, help="directory with vocab.json/train.txt/val.txt")
    ap.add_argument("--out", default=None, help="output directory (default t44k/model, or t44k/model_<config>)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the forward (needs triton; Windows: triton-windows<3.7 for torch 2.11)")
    ap.add_argument("--tf32", action="store_true", help="TF32 matmuls (integer-valued inputs up to 2048 stay exact)")
    ap.add_argument("--bench", action="store_true", help="speed test: stop after the training steps (no export)")
    ap.add_argument("--aux-frac", type=float, default=0.0, help="share of each batch from aux_train.txt (data/mix: Wikipedia)")
    ap.add_argument("--teacher", default=None, help="distil from a float teacher checkpoint (t44k/train_ta.py)")
    ap.add_argument("--kd", type=float, default=0.5, help="weight of the distillation loss (the rest: cross-entropy)")
    ap.add_argument("--kd-temp", type=float, default=1.0)
    ap.add_argument("--extra", default=None, help="extra corpus dir (train.txt/val.txt, same vocabulary), e.g. data/dialog64d")
    ap.add_argument("--extra-frac", type=float, default=0.0, help="share of each batch from --extra")
    ap.add_argument("--share", type=int, default=None, help="distinct weight sets for the L layers (cross-layer sharing)")
    ap.add_argument("--mix", nargs="+", default=None, metavar="DIR=FRAC",
                    help="train on several corpora at once (replaces --data/--aux-frac/--extra), e.g. "
                         "--mix data/mix64d=0.4 data/wiki64=0.15 data/dialog64f=0.45")
    ap.add_argument("--vocab", default=None, help="--mix: vocabulary file (default: the first DIR's vocab.json)")
    ap.add_argument("--resume", action="store_true", help="continue from OUT/ckpt.pt if it exists")
    ap.add_argument("--ckpt-every", type=int, default=2000, help="save OUT/ckpt.pt every N steps")
    a = ap.parse_args()
    S.configure(a.config)
    out_dir = a.out or os.path.join(HERE, "model" if a.config == "t44k" else f"model_{a.config}")
    torch.manual_seed(a.seed); random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    mix = None
    if a.mix:
        vocab, mix = load_mix(a.mix, a.vocab)
        train, val, aux = mix[0][2], mix[0][3], None
        dlg = [m for m in mix if m[4]]
        extra, extra_val = (dlg[0][2], dlg[0][3]) if dlg else (None, None)
        print("mix: " + ", ".join(f"{n} {f:.2f} ({len(t):,} seqs{', dialogue' if d else ''})" for n, f, t, _, d in mix))
    else:
        vocab, train, val, aux, _ = load_data(a.data, aux=True)
        if a.aux_frac and aux is None:
            raise SystemExit("--aux-frac needs aux_train.txt in --data")
        extra = extra_val = None
        if a.extra:
            extra, extra_val = load_extra(a.extra, vocab)
    teacher = None
    if a.teacher:
        from t44k.train_ta import load_ta
        teacher = load_ta(a.teacher, vocab, a.device)
    allowed = [S.EOS] + list(range(4, S.V))
    notallowed = torch.ones(S.V, dtype=torch.bool); notallowed[allowed] = False
    model = T44K(a.zero_frac, a.share).to(a.device)
    if a.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    val_eval = val[:8192]                                   # a fixed subset, evaluated in chunks
    log, t0, start = [], time.time(), 1
    os.makedirs(out_dir, exist_ok=True)
    ckpt, stop_file = os.path.join(out_dir, "ckpt.pt"), os.path.join(out_dir, "STOP")
    if a.resume and os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=a.device, weights_only=False)
        assert st["steps"] == a.steps and st["config"] == a.config, "checkpoint from a different run (steps / config)"
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        log, start, t0 = st["log"], st["step"] + 1, time.time() - st["elapsed"]
        torch.set_rng_state(st["rng"].cpu())               # map_location moved the RNG states too
        if st.get("cuda_rng") is not None and a.device.startswith("cuda"):
            torch.cuda.set_rng_state(st["cuda_rng"].cpu())
        print(f"resumed at step {st['step']} ({st['elapsed'] / 3600:.2f} h done)")
    fwd = torch.compile(model) if a.compile else model      # export() and the integer checks use the plain module

    def save_ckpt(step):
        st = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
              "steps": a.steps, "config": a.config, "log": log, "elapsed": time.time() - t0, "rng": torch.get_rng_state(),
              "cuda_rng": torch.cuda.get_rng_state() if a.device.startswith("cuda") else None}
        torch.save(st, ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)

    def progress(status, step, loss=None):
        el = time.time() - t0
        done = max(1, step - (start - 1))
        rate = (time.time() - t_run) / done
        json.dump({"status": status, "step": step, "steps": a.steps, "elapsed_s": round(el), "eta_s": round(rate * (a.steps - step)),
                   "train_loss": loss, "last_eval": log[-1] if log else None},
                  open(os.path.join(out_dir, "progress.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"device {a.device}, train {len(train)} seqs, val {len(val)}, vocab {len(vocab)}")

    def evaluate(store=None):
        store = val_eval if store is None else store
        model.eval()
        tot_loss = hit = n = 0
        with torch.no_grad():
            for k in range(0, len(store), 1024):
                xv, yv = batchify(store[k:k + 1024], a.device)
                lg = model(xv).masked_fill(notallowed.to(a.device), -1e9)
                m = yv != -100
                tot_loss += F.cross_entropy(lg.reshape(-1, S.V), yv.reshape(-1), ignore_index=-100, reduction="sum").item()
                hit += ((lg.argmax(-1) == yv) & m).sum().item(); n += m.sum().item()
        model.train()
        return tot_loss / n, hit / n

    train_dev = torch.from_numpy(train).to(a.device)        # int16 store on the GPU (~90 MB for T1M)
    aux_dev = torch.from_numpy(aux).to(a.device) if a.aux_frac and not mix else None
    extra_dev = torch.from_numpy(extra).to(a.device) if extra is not None and a.extra_frac and not mix else None
    mix_dev = [torch.from_numpy(m[2]).to(a.device) for m in mix] if mix else None
    na_dev = notallowed.to(a.device)
    t_run = time.time()
    for step in range(start, a.steps + 1):
        if mix:
            rows = mixed_batch_n(mix_dev, [m[1] for m in mix], a.batch, a.device)
        else:
            rows = mixed_batch(train_dev, aux_dev, a.aux_frac, a.batch, a.device, extra_dev, a.extra_frac)
        xb, yb, mb = batchify(rows, a.device, with_mask=True)
        lg = fwd(xb).masked_fill(na_dev, -1e9)
        loss = F.cross_entropy(lg.reshape(-1, S.V), yb.reshape(-1), ignore_index=-100)
        if teacher is not None:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device.startswith("cuda")):
                tl = teacher(xb).float().masked_fill(na_dev, -1e9)
            loss = (1 - a.kd) * loss + a.kd * kd_loss(lg, tl, mb.float(), a.kd_temp)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 500 == 0 or step == a.steps:
            vl, va = evaluate()
            e = {"step": step, "train_loss": round(loss.item(), 4), "val_loss": round(vl, 4), "val_acc": round(va, 4)}
            if mix:
                for n, _, _, v, _ in mix[1:]:
                    e[f"val_{n}"] = [round(x, 4) for x in evaluate(v[:4096])]
            if extra_val is not None:
                e["extra_val_loss"], e["extra_val_acc"] = (round(x, 4) for x in evaluate(extra_val[:8192]))
            log.append(e)
            print(f"step {step:6d}  train {loss.item():.3f}  val {vl:.3f}  val top-1 {va:.3f}"
                  + (f"  dialog {e['extra_val_loss']:.3f} / {e['extra_val_acc']:.3f}" if extra_val is not None else "")
                  + f"  ({time.time() - t0:.0f}s)", flush=True)
        if step % 100 == 0 or step == a.steps:
            progress("running", step, round(loss.item(), 4))
        if step % a.ckpt_every == 0 and step < a.steps:
            save_ckpt(step)
        if step % 100 == 0 and os.path.exists(stop_file) and step < a.steps:
            save_ckpt(step)
            os.remove(stop_file)
            progress("paused", step, round(loss.item(), 4))
            print(f"paused at step {step}: rerun with --resume to continue", flush=True)
            return

    if a.bench:
        return
    progress("exporting", a.steps)
    q = model.export(allowed)
    # calibrate the RMSNorm LUTs on training prompts (integer model), then measure integer accuracy
    n_cal, n_eval = (3000, 400) if S.D <= 32 else (2000, 400) if S.D <= 128 else (1000, 200)   # T1M: ~5 ms per integer step
    cal = random.Random(1).sample(range(len(train)), min(n_cal, len(train)))
    cal_seqs = [seq(train, i)[:S.T] for i in cal]
    if extra is not None:                                  # dialogue inputs reach the norms too
        cal_seqs += [seq(extra, i)[:S.T] for i in random.Random(2).sample(range(len(extra)), min(n_cal // 2, len(extra)))]
    S.calibrate(q, cal_seqs)
    im = S.IntModel(q)
    hit = tot = 0
    for i in range(min(n_eval, len(val))):
        s = seq(val, i)
        pred = im.teacher_forced(s[:-1][:S.T])
        tgt = s[1:][:S.T]
        ok = [(p, t) for p, t in zip(pred, tgt) if t != UNK]
        hit += sum(int(p == t) for p, t in ok); tot += len(ok)
    int_acc = hit / tot
    samples = []
    for s in (seq(val, i) for i in range(min(12, len(val)))):
        prompt = s[:3]
        out = im.generate(prompt, max_new=S.T - len(prompt))
        samples.append("".join(vocab[t] for t in prompt[1:]) + "→" + "".join("<EOS>" if t == S.EOS else vocab[t] for t in out))
    word = lambda ids: "".join("|" if t == S.EOS else vocab[t] for t in ids)
    for i in range(min(16, 0 if extra_val is None else len(extra_val))):   # dialogue: first turn + EOS -> reply
        s = seq(extra_val, i)
        prompt = s[:s.index(S.EOS, 1) + 1]
        out = im.generate(prompt, max_new=min(20, S.T - len(prompt)))
        samples.append("[对话] " + word(prompt[1:]) + " → " + word(out))
    json.dump(q, open(os.path.join(out_dir, "q.json"), "w"))
    json.dump({"args": vars(a), "log": log, "int_val_top1": round(int_acc, 4), "samples": samples,
               "constants": {k: q[k] for k in ("ATTDIV", "MO", "M2", "R")}, "vocab_size": len(vocab)},
              open(os.path.join(out_dir, "train_log.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    progress("done", a.steps)
    print(f"integer model val top-1 {int_acc:.3f} (float {log[-1]['val_acc']:.3f})")
    print("samples:", *samples, sep="\n  ")


if __name__ == "__main__":
    main()
