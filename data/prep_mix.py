"""Teacher corpus (+ Chinese Wikipedia as auxiliary data) -> a new 1024-character vocabulary and
training files for the distillation pipeline (data/gen_teacher.py -> t44k/train_ta.py -> t44k/train.py).

The vocabulary is chosen from the TEACHER text (the distribution the game model should produce):
<PAD> <BOS> <EOS> <UNK> + ，。！？、 + the 1015 most frequent characters of the teacher corpus.

teacher sentences (data/teacher2/raw.jsonl, optionally more folders):
  cleaned like prep_wiki.clean; only Chinese characters and ，。！？、; 6..31 characters; ends with
  。！？; no two punctuation marks in a row; at most MAX_UNK characters outside the vocabulary,
  written as '□' (input only, never a training target); NetEase lint; exact duplicates dropped;
  val = every request whose id % 50 == 0 (split by request, so no near-duplicate leaks)
wiki (data/wiki_raw.json): re-cut with prep_wiki.samples_of and re-filtered with the NEW vocabulary
  (the old data/wiki/train.txt already has the old vocabulary baked in)

  python data/prep_mix.py [--teacher data/teacher2 ...] [--out data/mix] [--no-wiki]
Outputs in --out: vocab.json, train.txt / val.txt (teacher), aux_train.txt / aux_val.txt (wiki),
stats.json (provenance)
"""
import sys, os, json, re, random, argparse
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import netease_lint
from prep_wiki import clean, samples_of, articles, SPECIALS, UNK_CH, PUNCT, OKCH, SOURCE as WIKI_SOURCE

MINLEN, MAXLEN = 6, 31   # one teacher sentence; --maxlen > 31 packs sentences into longer samples
END = "。！？"
MAX_UNK = 1            # teacher sentences
MAX_UNK_WIKI = 2       # as in prep_wiki


def teacher_rows(dirs):
    for d in dirs:
        for l in open(os.path.join(d, "raw.jsonl"), encoding="utf-8"):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:         # last line still being written by gen_teacher.py
                continue
            yield d, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", nargs="+", default=[os.path.join(HERE, "teacher2")])
    ap.add_argument("--out", default=os.path.join(HERE, "mix"))
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--no-wiki", action="store_true")
    ap.add_argument("--vocab-file", default=None, help="reuse this vocab.json instead of choosing a new one")
    ap.add_argument("--maxlen", type=int, default=31, help="sample length limit = context - 1 (64-context models: 63)")
    ap.add_argument("--wiki-only", action="store_true",
                    help="no teacher text: Wikipedia alone as train.txt / val.txt (one corpus for train.py --mix; needs --vocab-file)")
    ap.add_argument("--wiki-file", default=os.path.join(HERE, "wiki_raw.json"))
    a = ap.parse_args()
    import prep_wiki
    prep_wiki.MAXLEN = a.maxlen                          # samples_of() packs Wikipedia sentences up to this
    if a.wiki_only:
        assert a.vocab_file and not a.no_wiki, "--wiki-only needs --vocab-file"
        a.teacher = []

    rows = []
    for d, r in teacher_rows(a.teacher):
        s = clean(r["text"])
        rows.append((s, r.get("req", 0), d, r.get("model", "?")))
    freq = Counter(c for s, *_ in rows for c in s if "一" <= c <= "鿿")
    chars = list(PUNCT) + [c for c, _ in freq.most_common(a.vocab - len(SPECIALS) - len(PUNCT))]
    if a.vocab_file:                                     # keep an existing vocabulary (same token ids)
        chars = json.load(open(a.vocab_file, encoding="utf-8"))["vocab"][len(SPECIALS):]
    cs = set(chars)
    tot = sum(freq.values())

    seen, train, val, why = set(), [], [], Counter()
    for s, req, d, model in rows:
        if not s or not OKCH.match(s):          why["characters"] += 1; continue
        if not MINLEN <= len(s) <= MAXLEN:      why["length"] += 1; continue
        if s[-1] not in END:                    why["end"] += 1; continue
        if re.search(r"[，。！？、]{2}", s):     why["punct"] += 1; continue
        if sum(1 for c in s if c not in cs) > MAX_UNK: why["oov"] += 1; continue
        s = "".join(c if c in cs else UNK_CH for c in s)
        if netease_lint._pat.search(s):         why["sensitive"] += 1; continue
        if s in seen:                           why["duplicate"] += 1; continue
        seen.add(s)
        (val if req % 50 == 0 else train).append((req, s))
    if a.maxlen > MAXLEN:
        # longer context: consecutive sentences of the same request (same topics; a story in order)
        # packed into paragraphs of 1-4 sentences up to maxlen characters
        rng = random.Random(2)

        def pack(items):
            out, cur, k, last = [], "", rng.randint(1, 4), None
            for req, s in items:
                if cur and (req != last or len(cur) + len(s) > a.maxlen or k == 0):
                    out.append(cur); cur, k = "", rng.randint(1, 4)
                cur += s; k -= 1; last = req
            return out + ([cur] if cur else [])
        train, val = pack(train), pack(val)
    else:
        train, val = [s for _, s in train], [s for _, s in val]
    random.Random(0).shuffle(train)

    os.makedirs(a.out, exist_ok=True)
    models = Counter(m for *_, m in rows)
    teacher_src = {"generator": "data/gen_teacher.py", "platform": "api.deepseek.com", "models": dict(models),
                   "folders": [os.path.relpath(d, HERE) for d in a.teacher],
                   "terms": "DeepSeek Open Platform Terms: outputs may be used to train other models (distillation)"}
    vocab = SPECIALS + chars
    json.dump({"vocab": vocab, "source": {"teacher": teacher_src, "aux": WIKI_SOURCE}},
              open(os.path.join(a.out, "vocab.json"), "w", encoding="utf-8"), ensure_ascii=False)
    for name, lines in (("train.txt", train), ("val.txt", val)):
        if not a.wiki_only:
            open(os.path.join(a.out, name), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    st = {"teacher": teacher_src, "teacher_raw": len(rows), "teacher_train": len(train), "teacher_val": len(val),
          "teacher_rejected": dict(why), "teacher_distinct_chars": len(freq),
          "vocab_char_coverage_teacher": round(sum(freq[c] for c in cs) / max(1, tot), 4),
          "teacher_unk_share": round(sum(s.count(UNK_CH) for s in train) / max(1, sum(map(len, train))), 4),
          "teacher_mean_len": round(sum(map(len, train)) / max(1, len(train)), 2),
          "teacher_train_chars": sum(map(len, train))}
    print(json.dumps(st, ensure_ascii=False, indent=2))

    if not a.no_wiki:
        per_art = []
        for text in articles(a.wiki_file):
            k = []
            for s in samples_of(text):
                # same <UNK> density as the 31-character cut: longer samples may carry proportionally more
                if sum(1 for c in s if c not in cs) > MAX_UNK_WIKI * ((len(s) + 30) // 31) or s[-1] not in cs:
                    continue
                s = "".join(c if c in cs else UNK_CH for c in s)
                if not netease_lint._pat.search(s):
                    k.append(s)
            if k:
                per_art.append(k)
        random.Random(0).shuffle(per_art)
        n_val = max(1, len(per_art) // 50)
        wv = [s for x in per_art[:n_val] for s in x]
        wt = [s for x in per_art[n_val:] for s in x]
        random.Random(1).shuffle(wt)
        tn, vn = ("train.txt", "val.txt") if a.wiki_only else ("aux_train.txt", "aux_val.txt")
        open(os.path.join(a.out, tn), "w", encoding="utf-8").write("\n".join(wt) + "\n")
        open(os.path.join(a.out, vn), "w", encoding="utf-8").write("\n".join(wv) + "\n")
        st.update({"aux": WIKI_SOURCE, "aux_train": len(wt), "aux_val": len(wv), "aux_train_chars": sum(map(len, wt)),
                   "aux_unk_share": round(sum(s.count(UNK_CH) for s in wt) / max(1, sum(map(len, wt))), 4)})
        print({k: st[k] for k in ("aux_train", "aux_val", "aux_train_chars", "aux_unk_share")})
    json.dump(st, open(os.path.join(a.out, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
