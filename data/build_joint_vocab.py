"""Joint 1024-character vocabulary for the dialogue models: half teacher text (data/teacher2, the
written style), half human conversation (LCCC base, the spoken style). The teacher-only vocabulary
covered just 87.5% of LCCC characters and missed 啊 啦 呀 嗯 哦 嘛 啥 您 … (2026-10-03).

score(c) = 0.5 * share_in_teacher(c) + 0.5 * share_in_lccc(c); vocabulary = specials + ，。！？、 +
the 1015 best characters.

  python data/build_joint_vocab.py [--out data/mixd/vocab.json] [--lccc-dialogs 2000000]
"""
import os, sys, json, gzip, argparse
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prep_wiki import SPECIALS, PUNCT, clean


def cjk(s):
    return (c for c in s if "一" <= c <= "鿿")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "mixd", "vocab.json"))
    ap.add_argument("--lccc-dialogs", type=int, default=2000000)
    a = ap.parse_args()
    ft, fl = Counter(), Counter()
    for l in open(os.path.join(HERE, "teacher2", "raw.jsonl"), encoding="utf-8"):
        try:
            ft.update(cjk(clean(json.loads(l)["text"])))
        except json.JSONDecodeError:
            pass
    with gzip.open(os.path.join(HERE, "lccc", "lccc_base_train.jsonl.gz"), "rt", encoding="utf-8") as f:
        for i, l in enumerate(f):
            if i >= a.lccc_dialogs:
                break
            for t in json.loads(l):
                fl.update(cjk(t.replace(" ", "")))
    nt, nl = sum(ft.values()), sum(fl.values())
    score = Counter({c: 0.5 * ft[c] / nt + 0.5 * fl[c] / nl for c in set(ft) | set(fl)})
    chars = list(PUNCT) + [c for c, _ in score.most_common(1024 - len(SPECIALS) - len(PUNCT))]
    cs = set(chars)
    cov = lambda f, n: round(sum(x for c, x in f.items() if c in cs) / n, 4)
    old = set(json.load(open(os.path.join(HERE, "mix", "vocab.json"), encoding="utf-8"))["vocab"][4:])
    info = {"coverage_teacher": cov(ft, nt), "coverage_lccc": cov(fl, nl),
            "added_vs_teacher_vocab": "".join(c for c in chars if c not in old),
            "dropped_vs_teacher_vocab": "".join(c for c in sorted(old - cs))}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"vocab": SPECIALS + chars, "source": {"teacher": "data/teacher2 (DeepSeek, see data/mix64b/stats.json)",
               "dialogue": "LCCC base (thu-coai), research use", "rule": "0.5 teacher share + 0.5 LCCC share"},
               "stats": info}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(json.dumps(info, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
