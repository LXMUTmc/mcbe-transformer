"""Chinese Wikipedia -> T1M training text (character vocabulary, context 32 = BOS + 31 characters).

Source: 0xDing/wikipedia-cn-20230720-filtered (Hugging Face), Chinese Wikipedia dump of 2023-07-20,
filtered and converted to simplified Chinese by the dataset author. Licence CC BY-SA 3.0 (text by
Wikipedia contributors). Human-written text; no model outputs.

  python data/prep_wiki.py WIKI_JSON [--vocab 1024] [--out data/wiki] [--max-articles N]

- NFKC, ASCII ,!?;: -> Chinese punctuation; drop parenthetical notes （…） (…) (mostly foreign
  names and dates); drop quote/book marks but keep their text
- sentences end at 。！？ ; a sample is one sentence of 6..31 characters, or several consecutive
  short sentences of the same article packed up to 31 characters
- only Chinese characters and ，。！？、 are allowed; vocabulary = <PAD> <BOS> <EOS> <UNK> + the most
  frequent characters; samples with any character outside the vocabulary are dropped
- NetEase sensitive-word lint on every sample; train/val split by ARTICLE (98/2)
Outputs in --out: vocab.json, train.txt, val.txt, stats.json (with provenance and licence)
"""
import sys, os, json, re, random, unicodedata
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import netease_lint

SPECIALS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
UNK_CH = "□"          # stands for <UNK> in train/val text: input only, never a training target
MAX_UNK = 2           # samples with more out-of-vocabulary characters are dropped
PUNCT = "，。！？、"
MAXLEN, MINLEN = 31, 6
PAREN = re.compile(r"（[^（）]*）|\([^()]*\)")
DROP_MARKS = re.compile(r"[「」『』“”‘’《》〈〉\"'【】]")
SENT = re.compile(r"[^。！？]*[。！？]")
OKCH = re.compile(r"^[一-鿿，。！？、]+$")
SOURCE = {"dataset": "0xDing/wikipedia-cn-20230720-filtered", "dump": "zhwiki 2023-07-20",
          "licence": "CC BY-SA 3.0 (Wikipedia contributors)", "url": "https://huggingface.co/datasets/0xDing/wikipedia-cn-20230720-filtered"}


def articles(path, limit=None):
    with open(path, encoding="utf-8") as f:
        first = f.read(1); f.seek(0)
        rows = json.load(f) if first == "[" else (json.loads(l) for l in f if l.strip())
        for k, r in enumerate(rows):
            if limit and k >= limit:
                return
            yield r.get("completion") or r.get("text") or ""


def clean(text):
    s = unicodedata.normalize("NFKC", text)
    s = s.replace(",", "，").replace("!", "！").replace("?", "？").replace(";", "，").replace("；", "，")
    s = s.replace(":", "，").replace("：", "，").replace("．", "。")
    for _ in range(2):
        s = PAREN.sub("", s)
    s = DROP_MARKS.sub("", s)
    return re.sub(r"\s+", "", s)


CLAUSE = re.compile(r"[^，]*，|[^，]+$")


def samples_of(text):
    """whole sentences (MINLEN..MAXLEN) + packs of consecutive short sentences; a sentence longer
    than MAXLEN is cut at commas into packs of clauses (a pack that ends with '，' is not the end
    of a sentence: the trainer appends no EOS to it)"""
    sents = [x for x in SENT.findall(clean(text)) if OKCH.match(x)]
    out, cur = [], ""
    for x in sents:
        if len(x) > MAXLEN:
            if len(cur) >= MINLEN:
                out.append(cur)
            cur = ""
            for cl in CLAUSE.findall(x):
                if len(cl) > MAXLEN:                 # a clause too long for the context: drop it
                    if len(cur) >= MINLEN:
                        out.append(cur)
                    cur = ""
                elif len(cur) + len(cl) <= MAXLEN:
                    cur += cl
                else:
                    if len(cur) >= MINLEN:
                        out.append(cur)
                    cur = cl
            if len(cur) >= MINLEN:
                out.append(cur)
            cur = ""
            continue
        if len(cur) + len(x) <= MAXLEN:
            cur += x
        else:
            if len(cur) >= MINLEN:
                out.append(cur)
            cur = x
    if len(cur) >= MINLEN:
        out.append(cur)
    return out


def main(path, V=1024, out_dir=None, limit=None):
    out_dir = out_dir or os.path.join(HERE, "wiki")
    per_art, n_art, n_chars_raw = [], 0, 0
    for text in articles(path, limit):
        n_art += 1; n_chars_raw += len(text)
        s = samples_of(text)
        if s:
            per_art.append(s)
    freq = Counter(c for a in per_art for s in a for c in s)
    chars = list(PUNCT) + [c for c, _ in freq.most_common() if c not in PUNCT][:V - len(SPECIALS) - len(PUNCT)]
    cs = set(chars)
    kept_art, dropped = [], Counter()
    for a in per_art:
        k = []
        for s in a:
            n_oov = sum(1 for c in s if c not in cs)
            if n_oov > MAX_UNK or s[-1] not in cs:
                dropped["out of vocabulary"] += 1; continue
            s = "".join(c if c in cs else UNK_CH for c in s)
            try:
                netease_lint.check([s])
            except ValueError:
                dropped["sensitive"] += 1; continue
            k.append(s)
        if k:
            kept_art.append(k)
    random.Random(0).shuffle(kept_art)
    n_val = max(1, len(kept_art) // 50)
    val = [s for a in kept_art[:n_val] for s in a]
    train = [s for a in kept_art[n_val:] for s in a]
    os.makedirs(out_dir, exist_ok=True)
    vocab = SPECIALS + chars
    json.dump({"vocab": vocab, "source": SOURCE}, open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8"), ensure_ascii=False)
    open(os.path.join(out_dir, "train.txt"), "w", encoding="utf-8").write("\n".join(train) + "\n")
    open(os.path.join(out_dir, "val.txt"), "w", encoding="utf-8").write("\n".join(val) + "\n")
    total = sum(freq.values())
    stats = {"source": SOURCE, "articles": n_art, "raw_chars": n_chars_raw, "articles_with_samples": len(per_art),
             "candidate_samples": sum(len(a) for a in per_art), "distinct_chars": len(freq), "vocab": len(vocab),
             "vocab_char_coverage": round(sum(freq[c] for c in chars) / max(1, total), 4), "dropped": dict(dropped),
             "max_unk_per_sample": MAX_UNK, "unk_share": round(sum(s.count(UNK_CH) for s in train) / max(1, sum(map(len, train))), 4),
             "sentence_final_share": round(sum(s[-1] != "，" for s in train) / max(1, len(train)), 3),
             "train": len(train), "val": len(val), "train_chars": sum(map(len, train)),
             "mean_len": round(sum(map(len, train)) / max(1, len(train)), 2)}
    json.dump(stats, open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda k, d: (a[a.index(k) + 1] if k in a else d)
    files = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or not a[i - 1].startswith("--"))]
    main(files[0], int(opt("--vocab", 1024)), opt("--out", None), int(opt("--max-articles", 0)) or None)
