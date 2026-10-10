"""LCCC (human Chinese conversations, Weibo etc.) -> dialogue samples for the 64-context models.

Source: LCCC base (thu-coai, "A Large-Scale Chinese Short-Text Conversation Dataset", 2020),
https://github.com/thu-coai/CDial-GPT ; files from huggingface.co/datasets/silver/lccc
(lccc_base_{train,valid,test}.jsonl.gz). Repository licence MIT; the authors state the data is for
RESEARCH purposes only (user decision 2026-10-03: use it). Human-written; no model outputs.

Format: turns separated by '|' (encoded as <EOS> by t44k/train.py), e.g. "你好|你好呀" =
BOS 你 好 EOS 你 好 呀 EOS. After an EOS the other speaker talks; in the game the user's line
ends with EOS and the model answers until its EOS.

Cleaning: remove the per-character spaces; prep_wiki.clean (NFKC, ASCII punctuation -> Chinese,
drop brackets/quotes); utterances must be only Chinese characters and ，。！？、, 2..30 characters,
at most 1 character outside the vocabulary (written '□'), no NetEase-filtered word, no word of a
profanity list (the whole dialogue is dropped); a sample is the longest prefix of consecutive
turns (>= 2 turns) that fits into 63 characters including the separators.

  python data/prep_dialog.py [--vocab data/mix/vocab.json] [--out data/dialog64]
"""
import sys, os, re, gzip, json, random, argparse
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import netease_lint
from prep_wiki import clean, UNK_CH, OKCH

SEP = "|"
MAXLEN = 63
# dialogue is dropped when any turn contains one of these (residual insults in social-media text)
PROFANITY = ["傻逼", "傻b", "煞笔", "沙比", "妈的", "他妈", "你妈", "尼玛", "操你", "草你", "日你", "去死", "滚蛋", "贱人",
             "婊", "屌", "鸡巴", "逼", "屁眼", "王八蛋", "狗日", "智障", "脑残", "死全家", "约炮", "裸"]
_prof = re.compile("|".join(map(re.escape, PROFANITY)))
EMOTE = re.compile(r"\[[^\]]{1,8}\]")          # Weibo emoticons like [哈哈]


def utterance(s, cs):
    s = EMOTE.sub("", s.replace(" ", ""))
    s = clean(s)
    if not s or not OKCH.match(s) or not 2 <= len(s) <= 30:
        return None
    if sum(1 for c in s if c not in cs) > 1:
        return None
    return "".join(c if c in cs else UNK_CH for c in s)


def dialogue(turns, cs):
    out = []
    for t in turns:
        if _prof.search(t.replace(" ", "")):
            return None                                   # drop the whole dialogue
        u = utterance(t, cs)
        if u is None:
            break                                         # keep the clean prefix only
        if len(SEP.join(out + [u])) > MAXLEN:
            break
        out.append(u)
    if len(out) < 2:
        return None
    s = SEP.join(out)
    if netease_lint._pat.search(s):
        return None
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default=os.path.join(HERE, "mix", "vocab.json"))
    ap.add_argument("--src", default=os.path.join(HERE, "lccc"))
    ap.add_argument("--out", default=os.path.join(HERE, "dialog64"))
    a = ap.parse_args()
    vocab = json.load(open(a.vocab, encoding="utf-8"))["vocab"]
    cs = set(vocab[4:])
    stats, out = Counter(), {}
    for split, fname in (("train", "lccc_base_train.jsonl.gz"), ("val", "lccc_base_valid.jsonl.gz")):
        kept, seen = [], set()
        with gzip.open(os.path.join(a.src, fname), "rt", encoding="utf-8") as f:
            for line in f:
                stats[split + "_raw"] += 1
                d = dialogue(json.loads(line), cs)
                if d is None or d in seen:
                    continue
                seen.add(d); kept.append(d)
        random.Random(0).shuffle(kept)
        out[split] = kept
        stats[split] = len(kept)
        stats[split + "_chars"] = sum(len(s.replace(SEP, "")) for s in kept)
        stats[split + "_turns"] = sum(s.count(SEP) + 1 for s in kept)
    os.makedirs(a.out, exist_ok=True)
    for split, kept in out.items():
        open(os.path.join(a.out, split + ".txt"), "w", encoding="utf-8").write("\n".join(kept) + "\n")
    info = dict(stats)
    info["source"] = {"dataset": "LCCC base (thu-coai)", "files": "huggingface.co/datasets/silver/lccc",
                      "licence": "MIT repository; authors: research use only", "paper": "arXiv:2008.03946"}
    json.dump(info, open(os.path.join(a.out, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    for s in out["val"][:12]:
        print("   ", s)


if __name__ == "__main__":
    main()
