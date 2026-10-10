"""Your own Chinese text -> training samples for the 64-context models (one corpus directory for
t44k/train.py --mix).

Two kinds of input (UTF-8 or GB18030 .txt files):
  text     prose (novels, articles, notes): cleaned like the Wikipedia data (data/prep_wiki.clean),
           cut into sentences and packed into samples of up to 63 characters
  dialog   conversations: either one dialogue per line with turns separated by '|'
           (你好|你好呀|在干嘛), or one turn per line with a blank line between dialogues; a speaker
           prefix such as "A：" or "张三:" at the start of a turn is removed. A sample is the longest
           run of consecutive turns that fits into 63 characters (at least 2 turns).
Only Chinese characters and ，。！？、 are kept. Characters outside the vocabulary become '□' (input
only, never generated); samples with more than --max-unk of them are dropped.

  python data/prep_text.py FILE [FILE ...] --kind text|dialog --out data/mytext [--vocab data/mixd/vocab.json]
  python data/prep_text.py FILE ... --kind text --out data/mytext --new-vocab
      --new-vocab: build a 1024-character vocabulary from THIS text (every other corpus mixed in must
      then be prepared with the same vocab.json; the released weights cannot be reused)
Outputs in --out: train.txt, val.txt (every 50th sample), vocab.json, stats.json
"""
import os, re, sys, json, argparse
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import netease_lint
import prep_wiki as W

MAXLEN = 63
SEP = "|"
SPEAKER = re.compile(r"^[^，。！？：:|]{1,8}[：:]")
TURN_OK = re.compile(r"^[一-鿿，。！？、]+$")


def read_text(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    raise SystemExit(f"{path}: not UTF-8 or GB18030 text")


def text_samples(text):
    W.MAXLEN = MAXLEN
    out = []
    for para in re.split(r"\n\s*\n|\r?\n", text):
        out += W.samples_of(para)
    return out


def dialogues(text):
    """lists of raw turns"""
    lines = text.splitlines()
    if sum(1 for l in lines if SEP in l) > len(lines) * 0.3:
        return [l.split(SEP) for l in lines if SEP in l]
    out, cur = [], []
    for l in lines:
        if l.strip():
            cur.append(l)
        elif cur:
            out.append(cur); cur = []
    if cur:
        out.append(cur)
    return out


def turn(s):
    s = SPEAKER.sub("", s.strip())
    s = W.clean(s)
    return s if s and TURN_OK.match(s) and len(s) <= 30 else None


def dialog_samples(text):
    out = []
    for d in dialogues(text):
        turns = [turn(t) for t in d]
        run = []
        for t in turns + [None]:
            if t is None:                      # a bad turn breaks the dialogue: flush what fits
                while len(run) >= 2:
                    k = 2
                    while k < len(run) and len(SEP.join(run[:k + 1])) <= MAXLEN:
                        k += 1
                    if len(SEP.join(run[:k])) <= MAXLEN:
                        out.append(SEP.join(run[:k]))
                    run = run[k:]
                run = []
                continue
            run.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--kind", choices=("text", "dialog"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", default=os.path.join(HERE, "mixd", "vocab.json"))
    ap.add_argument("--new-vocab", action="store_true")
    ap.add_argument("--max-unk", type=int, default=1)
    a = ap.parse_args()
    samples = []
    for f in a.files:
        t = read_text(f)
        samples += text_samples(t) if a.kind == "text" else dialog_samples(t)
    samples = list(dict.fromkeys(samples))                 # exact duplicates
    if a.new_vocab:
        freq = Counter(c for s in samples for c in s if c != SEP)
        chars = list(W.PUNCT) + [c for c, _ in freq.most_common() if c not in W.PUNCT][:1024 - len(W.SPECIALS) - len(W.PUNCT)]
        vocab = W.SPECIALS + chars
        vocab += [f"<pad{i}>" for i in range(1024 - len(vocab))]   # tiny corpora: unused slots keep V = 1024
    else:
        vocab = json.load(open(a.vocab, encoding="utf-8"))["vocab"]
    cs = set(vocab)
    kept, st = [], Counter()
    for s in samples:
        n_unk = sum(1 for c in s if c != SEP and c not in cs)
        if n_unk > a.max_unk:
            st["too many characters outside the vocabulary"] += 1; continue
        s = "".join(c if c == SEP or c in cs else W.UNK_CH for c in s)
        if netease_lint._pat.search(s):                   # the game would swallow it as a prompt / reply
            st["NetEase filtered word"] += 1; continue
        kept.append(s)
    os.makedirs(a.out, exist_ok=True)
    val = kept[::50]
    train = [s for i, s in enumerate(kept) if i % 50]
    for name, rows in (("train", train), ("val", val)):
        open(os.path.join(a.out, name + ".txt"), "w", encoding="utf-8", newline="\n").write("\n".join(rows) + ("\n" if rows else ""))
    json.dump({"vocab": vocab}, open(os.path.join(a.out, "vocab.json"), "w", encoding="utf-8"), ensure_ascii=False)
    chars = Counter(c for s in kept for c in s if c != SEP)
    stats = {"source": [os.path.basename(f) for f in a.files], "kind": a.kind, "samples": len(kept), "train": len(train),
             "val": len(val), "characters": sum(chars.values()), "unk_share": round(chars[W.UNK_CH] / max(1, sum(chars.values())), 4),
             "dropped": dict(st), "new_vocab": a.new_vocab}
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False))
    if len(train) < 1000:
        print("warning: fewer than 1000 samples -- the model will mostly learn the other corpora (or overfit)")


if __name__ == "__main__":
    main()
