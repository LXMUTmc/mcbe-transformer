"""Float "teaching assistant" for distillation: a plain character-level Transformer with the SAME
vocabulary and context as the game model (data/mix), trained on the teacher corpus (+ Wikipedia).
The DeepSeek teacher only gives text (its probabilities are over its own tokens, not our 1024
characters); this model turns that text into full next-character distributions the ternary
student can learn from (t44k/train.py --teacher), and is small enough to run alongside it.

  python -m t44k.train_ta --data data/mix [--size m] [--steps 60000] [--aux-frac 0.3] [--compile]
Output: t44k/ta_<size>/ta.pt (config, weights, vocabulary hash) and train_log.json
Pre-LN, GELU, learned positions, tied head, bf16 autocast. Sizes: s 6.6M, m 25M, l 57M parameters.
"""
import argparse, hashlib, json, math, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from t44k import spec as S
from t44k.train import load_data, batchify, mixed_batch

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = {"s": dict(d=256, L=6, H=4), "m": dict(d=512, L=8, H=8), "l": dict(d=768, L=8, H=12)}


def vocab_hash(vocab):
    return hashlib.sha256("\n".join(vocab).encode("utf-8")).hexdigest()[:16]


class Block(nn.Module):
    def __init__(self, d, H, drop):
        super().__init__()
        self.H, self.drop = H, drop
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.f1, self.f2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.H, D // self.H).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0.0)
        x = x + F.dropout(self.o(a.transpose(1, 2).reshape(B, T, D)), self.drop, self.training)
        return x + F.dropout(self.f2(F.gelu(self.f1(self.n2(x)))), self.drop, self.training)


class TA(nn.Module):
    def __init__(self, V, T, d, L, H, drop=0.1):
        super().__init__()
        self.cfg = dict(V=V, T=T, d=d, L=L, H=H, drop=drop)
        self.E, self.P = nn.Embedding(V, d), nn.Parameter(torch.zeros(T, d))
        self.blocks = nn.ModuleList([Block(d, H, drop) for _ in range(L)])
        self.nf = nn.LayerNorm(d)
        nn.init.normal_(self.E.weight, std=0.02); nn.init.normal_(self.P, std=0.02)
        for b in self.blocks:                                   # GPT-2 style scaled residual init
            nn.init.normal_(b.o.weight, std=0.02 / math.sqrt(2 * L)); nn.init.normal_(b.f2.weight, std=0.02 / math.sqrt(2 * L))

    def forward(self, tok):
        x = self.E(tok) + self.P[:tok.shape[1]]
        for b in self.blocks:
            x = b(x)
        return self.nf(x) @ self.E.weight.t()


def load_ta(path, vocab, device):
    ck = torch.load(path, map_location=device)
    assert ck["vocab_hash"] == vocab_hash(vocab), "teacher was trained with another vocabulary"
    m = TA(**ck["config"]).to(device)
    m.load_state_dict(ck["state"]); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(HERE), "data", "mix"))
    ap.add_argument("--size", default="m", choices=SIZES)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--aux-frac", type=float, default=0.3, help="share of each batch from Wikipedia (aux_train.txt)")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default="t1m", help="vocabulary size / context of the students (t1m: T=32, t1m64: T=64)")
    ap.add_argument("--extra", default=None, help="extra corpus dir (e.g. data/dialog64d), same vocabulary")
    ap.add_argument("--extra-frac", type=float, default=0.0)
    a = ap.parse_args()
    S.configure(a.config)                                  # V=1024 and the students' context T
    torch.manual_seed(a.seed); random.seed(a.seed)
    out_dir = a.out or os.path.join(HERE, f"ta_{a.size}")
    vocab, train, val, aux, aux_val = load_data(a.data, aux=True)
    extra = extra_val = None
    if a.extra:
        from t44k.train import load_extra
        extra, extra_val = load_extra(a.extra, vocab)
    if aux is None: a.aux_frac = 0.0
    model = TA(S.V, S.T, drop=a.dropout, **SIZES[a.size]).to(a.device)
    n_par = sum(p.numel() for p in model.parameters())
    fwd = torch.compile(model) if a.compile else model
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    other = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.03)
    allowed = [S.EOS] + list(range(4, S.V))
    na = torch.ones(S.V, dtype=torch.bool); na[allowed] = False; na = na.to(a.device)
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=a.device.startswith("cuda"))
    print(f"TA-{a.size}: {n_par / 1e6:.1f}M parameters; teacher train {len(train)}, val {len(val)}; "
          f"aux {0 if aux is None else len(aux)} (frac {a.aux_frac})")

    def evaluate(store, n=8192):
        model.eval(); tl = hit = cnt = 0
        with torch.no_grad(), torch.autocast(**amp):
            for k in range(0, min(n, len(store)), 1024):
                x, y = batchify(store[k:k + 1024], a.device)
                lg = model(x).float().masked_fill(na, -1e9)
                m = y != -100
                tl += F.cross_entropy(lg.reshape(-1, S.V), y.reshape(-1), ignore_index=-100, reduction="sum").item()
                hit += ((lg.argmax(-1) == y) & m).sum().item(); cnt += m.sum().item()
        model.train()
        return tl / cnt, hit / cnt

    tr = torch.from_numpy(train).to(a.device)
    ax = torch.from_numpy(aux).to(a.device) if a.aux_frac else None
    ex = torch.from_numpy(extra).to(a.device) if extra is not None and a.extra_frac else None
    log, t0, best = [], time.time(), float("inf")
    os.makedirs(out_dir, exist_ok=True)

    def save():
        torch.save({"config": model.cfg, "state": model.state_dict(), "vocab_hash": vocab_hash(vocab), "step": step},
                   os.path.join(out_dir, "ta.pt"))
    for step in range(1, a.steps + 1):
        x, y = batchify(mixed_batch(tr, ax, a.aux_frac, a.batch, a.device, ex, a.extra_frac), a.device)
        with torch.autocast(**amp):
            lg = fwd(x).float().masked_fill(na, -1e9)
            loss = F.cross_entropy(lg.reshape(-1, S.V), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 1000 == 0 or step == a.steps:
            vl, va = evaluate(val)
            e = {"step": step, "train_loss": round(loss.item(), 4), "val_loss": round(vl, 4), "val_acc": round(va, 4)}
            if aux_val is not None:
                e["aux_val_loss"], e["aux_val_acc"] = (round(v, 4) for v in evaluate(aux_val))
            log.append(e)
            mark = ""
            if vl < best:                                   # keep the best checkpoint (teacher val loss)
                best, mark = vl, "  *saved"
                save()
            print(f"step {step:6d}  train {loss.item():.3f}  val {vl:.3f} / top-1 {va:.3f}"
                  + (f"  wiki {e['aux_val_loss']:.3f} / {e['aux_val_acc']:.3f}" if aux_val is not None else "")
                  + f"  ({time.time() - t0:.0f}s){mark}", flush=True)
            json.dump({"args": vars(a), "params": n_par, "best_val_loss": best, "log": log},
                      open(os.path.join(out_dir, "train_log.json"), "w"), indent=2)
    print("best teacher val loss", round(best, 4), "->", os.path.join(out_dir, "ta.pt"))


if __name__ == "__main__":
    main()
