"""Down-sample generic ("dull") replies in the dialogue data (data/prep_dialog.py output).

Small chat models trained on social-media conversations collapse to safe replies (谢谢 / 哈哈 / 是的 /
我也是 …): measured on T4M/T7M/T10M 2026-10-04, about half of the greedy replies were "我也是"-like.
A dialogue whose REPLY turns (turn 2 onwards) contain a generic reply is kept with probability
--keep. Generic = a normalised reply (trailing punctuation removed) seen at least --min-count
times in the data, or a reply of at most 6 characters starting with 我也. A turn with one character
repeated 4+ times (哈哈哈哈…) ends the dialogue (the clean prefix is kept if it has >= 2 turns).

  python data/dedull_dialog.py [--src data/dialog64d] [--out data/dialog64e] [--min-count 200] [--keep 0.1]
"""
import os, re, json, random, shutil, argparse
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
norm = lambda u: re.sub(r"[，。！？、]+$", "", u)
REPEAT = re.compile(r"(.)\1{3,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "dialog64d"))
    ap.add_argument("--out", default=os.path.join(HERE, "dialog64e"))
    ap.add_argument("--min-count", type=int, default=200)
    ap.add_argument("--keep", type=float, default=0.1)
    a = ap.parse_args()
    rows = [l.rstrip("\n") for l in open(os.path.join(a.src, "train.txt"), encoding="utf-8") if l.strip()]
    cnt = Counter(norm(u) for r in rows for u in r.split("|")[1:])
    generic = {u for u, x in cnt.items() if x >= a.min_count}
    is_generic = lambda u: norm(u) in generic or (norm(u).startswith("我也") and len(norm(u)) <= 6)
    rng = random.Random(0)
    st = Counter()

    def fix(r):
        t = r.split("|")
        cut = next((i for i, u in enumerate(t) if REPEAT.search(u)), len(t))
        if cut < len(t):
            st["repeat_cut"] += 1
            t = t[:cut]
        if len(t) < 2:
            return None
        if any(is_generic(u) for u in t[1:]):
            st["generic"] += 1
            if rng.random() >= a.keep:
                return None
        return "|".join(t)

    os.makedirs(a.out, exist_ok=True)
    for split in ("train", "val"):
        src = rows if split == "train" else [l.rstrip("\n") for l in open(os.path.join(a.src, "val.txt"), encoding="utf-8") if l.strip()]
        out = [x for x in (fix(r) for r in src) if x]
        open(os.path.join(a.out, split + ".txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
        st[split + "_in"], st[split + "_out"] = len(src), len(out)
        st[split + "_chars"] = sum(len(x.replace("|", "")) for x in out)
    shutil.copy(os.path.join(a.src, "vocab.json"), os.path.join(a.out, "vocab.json"))
    info = dict(st, generic_forms=len(generic), min_count=a.min_count, keep=a.keep, source=a.src)
    json.dump(info, open(os.path.join(a.out, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
