"""Mine question -> answer pairs from the cleaned LCCC dialogues (data/dialog64d) and keep the ones
whose answer really depends on the question.

Candidates: adjacent turns (q, a) where q is a question (ends with ？ / 吗 / 呢, or contains 什么,
怎么, 为什么, 为啥, 哪, 多少, 谁, 咋, 是不是, 有没有, 能不能, 会不会 ...).
  dropped : generic answers (a reply seen >= 200 times, see dedull_dialog.py) and short 哈哈… /
            我也… answers
  idk     : answers that say "I don't know" (不知道 / 不清楚 / 不晓得 …) are kept apart (honest replies)
Relevance (pointwise mutual information, scored by a float model t44k/ta_dialog trained on the
same vocabulary):  pmi = mean over the answer's tokens (and its <EOS>) of
  log p(token | <BOS> q <EOS> answer-so-far) - log p(token | <BOS> <EOS> answer-so-far)
i.e. how much more likely the answer is BECAUSE of the question. Kept: pmi above the --keep
quantile of the candidates and > 0; idk pairs are kept when pmi > 0 (capped at --idk-share of
the output).

  python data/mine_qa.py [--keep 0.6] [--out data/qa_lccc]
Output: train.txt / val.txt as "q|a" lines (turns separated by '|', like prep_dialog.py), stats.json
"""
import os, re, sys, json, random, argparse
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from t44k import spec as S
from t44k.train_ta import load_ta

Q = re.compile(r"[？?]$|吗[，。！]*$|呢[，。！]*$|什么|怎么|为什么|为啥|干嘛|哪|几个|几点|多少|谁|咋|是不是|有没有|能不能|会不会|要不要|好不好|对不对")
IDK = re.compile(r"^(我也)?(不知道|不清楚|不晓得|没听说|不太清楚|不懂|不确定)")
norm = lambda u: re.sub(r"[，。！？、]+$", "", u)


def candidates(rows, generic):
    out, st = [], Counter()
    for r in rows:
        t = r.split("|")
        for i in range(len(t) - 1):
            q, a = t[i], t[i + 1]
            if not Q.search(q):
                continue
            na = norm(a)
            if IDK.match(na):
                out.append((q, a, "idk")); st["idk"] += 1
            elif na in generic or ((na.startswith("哈哈") or na.startswith("我也")) and len(na) <= 6):
                st["generic"] += 1
            else:
                out.append((q, a, "qa")); st["qa"] += 1
    return out, st


@torch.no_grad()
def pmi(model, pairs, idx, dev, bs=512):
    """mean log-prob gain of the answer tokens given the question"""
    def enc(q, a):
        cond = [S.BOS] + [idx[c] for c in q] + [S.EOS]
        ans = [idx[c] for c in a] + [S.EOS]
        return cond, ans
    out = np.zeros(len(pairs), np.float32)
    for k in range(0, len(pairs), bs):
        chunk = pairs[k:k + bs]
        rows_c, rows_u, spans_c, spans_u = [], [], [], []
        for q, a, _ in chunk:
            cond, ans = enc(q, a)
            sc = (cond + ans)[:S.T + 1]
            su = ([S.BOS, S.EOS] + ans)[:S.T + 1]
            rows_c.append(sc); spans_c.append((len(cond), len(sc)))
            rows_u.append(su); spans_u.append((2, len(su)))

        def score(rows, spans):
            L = max(len(r) for r in rows)
            x = torch.zeros(len(rows), L - 1, dtype=torch.long)
            y = torch.zeros(len(rows), L - 1, dtype=torch.long)
            for j, r in enumerate(rows):
                x[j, :len(r) - 1] = torch.tensor(r[:-1]); y[j, :len(r) - 1] = torch.tensor(r[1:])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                lp = F.log_softmax(model(x.to(dev)).float(), -1)
            g = lp.gather(-1, y.to(dev).unsqueeze(-1)).squeeze(-1).cpu().numpy()
            return [g[j, s - 1:e - 1] for j, (s, e) in enumerate(spans)]   # predictions of the answer tokens
        gc, gu = score(rows_c, spans_c), score(rows_u, spans_u)
        for j in range(len(chunk)):
            n = min(len(gc[j]), len(gu[j]))
            out[k + j] = float(np.mean(gc[j][:n] - gu[j][:n])) if n else -9.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "dialog64d"))
    ap.add_argument("--out", default=os.path.join(HERE, "qa_lccc"))
    ap.add_argument("--ta", default=os.path.join(os.path.dirname(HERE), "t44k", "ta_dialog", "ta.pt"))
    ap.add_argument("--keep", type=float, default=0.6, help="drop the candidates below this pmi quantile")
    ap.add_argument("--idk-share", type=float, default=0.05)
    a = ap.parse_args()
    S.configure("t1m64")
    vocab = json.load(open(os.path.join(a.src, "vocab.json"), encoding="utf-8"))["vocab"]
    idx = {c: i for i, c in enumerate(vocab)}
    idx["□"] = 3
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_ta(a.ta, vocab, dev)
    rows = [l.rstrip("\n") for l in open(os.path.join(a.src, "train.txt"), encoding="utf-8") if l.strip()]
    cnt = Counter(norm(u) for r in rows for u in r.split("|")[1:])
    generic = {u for u, x in cnt.items() if x >= 200}
    info = {}
    for split in ("train", "val"):
        src = rows if split == "train" else [l.rstrip("\n") for l in open(os.path.join(a.src, "val.txt"), encoding="utf-8") if l.strip()]
        cands, st = candidates(src, generic)
        scores = pmi(model, cands, idx, dev)
        qa = [(c, s) for c, s in zip(cands, scores) if c[2] == "qa"]
        thr = float(np.quantile([s for _, s in qa], a.keep)) if qa else 0.0
        kept = [c for c, s in qa if s > max(thr, 0.0)]
        idk = [c for c, s in zip(cands, scores) if c[2] == "idk" and s > 0]
        random.Random(1).shuffle(idk)
        idk = idk[:int(len(kept) * a.idk_share / (1 - a.idk_share))]
        lines = sorted({f"{q}|{ans}" for q, ans, _ in kept + idk})
        random.Random(0).shuffle(lines)
        os.makedirs(a.out, exist_ok=True)
        open(os.path.join(a.out, split + ".txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
        info[split] = dict(st, candidates=len(cands), pmi_threshold=round(thr, 3), kept_qa=len(kept), kept_idk=len(idk), lines=len(lines))
        if split == "train":
            order = np.argsort([s for _, s in qa])
            info["top"] = [f"{qa[i][0][0]} → {qa[i][0][1]}  ({qa[i][1]:.2f})" for i in order[::-1][:15]]
            info["cut"] = [f"{qa[i][0][0]} → {qa[i][0][1]}  ({qa[i][1]:.2f})" for i in order[int(len(order) * a.keep) - 6:int(len(order) * a.keep) + 6]]
            info["bottom"] = [f"{qa[i][0][0]} → {qa[i][0][1]}  ({qa[i][1]:.2f})" for i in order[:8]]
    import shutil
    shutil.copy(os.path.join(a.src, "vocab.json"), os.path.join(a.out, "vocab.json"))
    info["source"] = "LCCC base via data/dialog64d (human dialogues; research use), relevance by t44k/ta_dialog"
    json.dump(info, open(os.path.join(a.out, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(info, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
