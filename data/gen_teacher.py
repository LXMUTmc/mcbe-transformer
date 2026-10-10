"""Teacher corpus for T1M-class students: short general Chinese sentences written by the DeepSeek
API (open-weight model family, MIT weights; DeepSeek Open Platform Terms allow using outputs to
train other models, e.g. distillation -- check the current terms yourself).

Data-source rules (DATA_AND_MODEL.md): never Claude/GPT/Gemini outputs; every line keeps its provenance.
Nothing written by Claude becomes training text: the prompt holds NO example sentences, only
instructions, the allowed character list and topic words. Topic words are taken from Chinese
Wikipedia article subjects (human-written, CC BY-SA 3.0), see --make-topics.

The student: character vocabulary of 1024 (data/wiki/vocab.json), context 32 = BOS + 31
characters, so we ask for complete sentences of 8..30 characters using only the vocabulary.

  python data/gen_teacher.py --make-topics            # once: data/teacher/topics.txt from wiki_raw.json
  python data/gen_teacher.py --dry-run                # show prompts, no API calls
  $env:DEEPSEEK_API_KEY = "sk-..."                    # never commit or print the key
  python data/gen_teacher.py --sentences 10000        # trial: stop after ~10K raw sentences
  python data/gen_teacher.py --stats                  # usable rate, tokens, estimated cost
  round 2 (no character list in the prompt, longer sentences; vocabulary chosen afterwards):
  python data/gen_teacher.py --dir data/teacher2 --sentences 10000 --workers 4
  python data/gen_teacher.py --dir data/teacher2 --stats --own-vocab
Round 1 (data/teacher, wiki character list in the prompt): 56% usable, mean 12.5 characters.
Resumable (requests are numbered; a rerun continues). Standard library only.
"""
import argparse, json, os, random, re, sys, time, threading, urllib.request, urllib.error
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import netease_lint

API_URL = "https://api.deepseek.com/chat/completions"
PLATFORM = "api.deepseek.com"
OUT_DIR = os.path.join(HERE, "teacher")
TOPICS = os.path.join(OUT_DIR, "topics.txt")
VOCAB = os.path.join(HERE, "wiki", "vocab.json")
PUNCT = "，。！？、"
END = "。！？"
MINLEN, MAXLEN = 6, 31

# style descriptions only, deliberately WITHOUT example sentences
STYLES = {
    "fact":     "写百科常识式的陈述句，讲清楚是什么、在哪里、有什么特点或用途，内容要准确。",
    "daily":    "写日常生活里的叙述句，讲人们平常做的事、看到的事。",
    "describe": "写描写句，描写景物、动物、天气、地方或人的样子和感受。",
    "story":    "写一个能连起来读的小故事，每句单独成立，按事情发生的顺序写。",
    "talk":     "写人们说话时会说的完整句子，口语化，可以是陈述、感叹或疑问。",
    "explain":  "写解释原因或说明道理的句子，例如为什么会这样、怎样做比较好。",
}
STYLE_W = {"fact": 3, "daily": 2, "describe": 2, "story": 2, "talk": 1, "explain": 1}


def load_vocab():
    v = json.load(open(VOCAB, encoding="utf-8"))["vocab"]
    return [c for c in v[4:] if c != "□"]


def system_prompt(chars=None):
    # fixed for every request -> served from the API's prefix cache.
    # chars=None (default since round 2): no character list, the student's vocabulary is chosen
    # afterwards from this corpus; with a list the teacher bent sentences around missing everyday
    # characters (round 1: 43% rejected, awkward phrasing)
    if chars:
        han = "".join(c for c in chars if c not in PUNCT)
        rule = ("它只认识下面这些汉字，所以你写的每一个字都必须来自这个列表；列表外的字一律不用，换个说法用列表里的字表达。\n"
                f"允许的汉字：{han}\n")
    else:
        rule = "它只认识最常用的简体汉字，所以尽量用常用字，少用生僻字、专有名词里的罕见字。\n"
    return ("你为一个很小的中文语言模型编写训练语料。" + rule +
            "允许的标点：，。！？、（不要用其他标点、引号、书名号、括号、数字、字母）。\n"
            "内容要求：通顺、自然、正确的现代汉语，不涉及政治、时事、真实在世人物的评价、广告、网址和个人信息。")


def user_prompt(topics, style, n):
    return (f"话题词：{'、'.join(topics)}。\n"
            f"{STYLES[style]}\n"
            f"要求：\n"
            f"1. 写 {n} 句，每句围绕至少一个话题词，或与它们相关的事物；句子之间不要重复，用词尽量多样。\n"
            f"2. 每句是完整的句子，以。！？之一结尾；长度 12 到 30 个字（标点也算），一半以上在 18 字以上，"
            f"可以用逗号连接两三个分句。\n"
            f"3. 只用常用简体汉字和允许的标点。\n"
            f"只输出一个 JSON 字符串数组，不要任何其他文字。")


# ---------------------------------------------------------------- topics from Wikipedia subjects
SUBJ = re.compile(r"^([一-鿿]{2,8})(?:（[^）]*）|\([^)]*\))?(?:是|为|，)")


def make_topics(raw=os.path.join(HERE, "wiki_raw.json"), out=os.path.join(OUT_DIR, "topics.txt"), min_count=30):
    cs = set(load_vocab())
    rows = json.load(open(raw, encoding="utf-8"))
    seen, topics = set(), []
    for r in rows:
        text = (r.get("completion") or "").strip()
        first = text.split("\n", 1)[0]
        cands = []
        if 2 <= len(first) <= 8 and re.fullmatch(r"[一-鿿]+", first):
            cands.append(first)                                   # title line
        m = SUBJ.match(first)
        if m:
            cands.append(m.group(1))                              # subject of "X是……"
        for t in cands:
            if t not in seen and all(c in cs for c in t):
                seen.add(t); topics.append(t)
    del rows
    topics = [t for t in topics if not netease_lint._pat.search(t) and t[0] not in "是为"]
    # keep common concepts, not obscure names: occurrences in the whole training text
    text = open(os.path.join(HERE, "wiki", "train.txt"), encoding="utf-8").read()
    freq = {t: text.count(t) for t in topics}
    kept = sorted((t for t in topics if freq[t] >= min_count), key=lambda t: -freq[t])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8").write("\n".join(kept) + "\n")
    print(f"{len(topics)} subject words -> {len(kept)} with >= {min_count} occurrences -> {out}")


# ---------------------------------------------------------------- API
def call_api(key, model, messages, temperature, timeout=180):
    body = json.dumps({"model": model, "messages": messages, "temperature": temperature,
                       "thinking": {"type": "disabled"}, "max_tokens": 4096, "stream": False}).encode("utf-8")
    req = urllib.request.Request(API_URL, data=body, method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def parse_lines(text):
    text = text.strip()
    s, e = text.find("["), text.rfind("]")
    if s >= 0 and e > s:
        try:
            arr = json.loads(text[s:e + 1])
            return [x.strip() for x in arr if isinstance(x, str) and x.strip()]
        except json.JSONDecodeError:
            pass
    return [x.strip() for x in re.findall(r'"([^"\n]{1,60})"', text) if x.strip()]


def task(i, topics, seed, n):
    """request i is fully determined by (seed, i): reruns and parallel workers agree"""
    r = random.Random(seed * 1000003 + i)
    style = r.choices(list(STYLE_W), weights=list(STYLE_W.values()))[0]
    pick = set()
    while len(pick) < 3:                       # topics are sorted by frequency: favour common ones
        pick.add(topics[int(len(topics) * r.random() ** 2)])
    return {"i": i, "topics": sorted(pick, key=topics.index), "style": style, "n": n,
            "temperature": round(r.uniform(0.9, 1.2), 2)}


def run(a):
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key and not a.dry_run:
        sys.exit("set the DEEPSEEK_API_KEY environment variable first (never put the key in a file)")
    topics = [t for t in open(TOPICS, encoding="utf-8").read().split() if t]
    sysmsg = system_prompt(load_vocab() if a.vocab_prompt else None)
    if a.dry_run:
        print("SYSTEM:", sysmsg[:200], "...", f"({len(sysmsg)} chars)")
        for i in range(3):
            t = task(i, topics, a.seed, a.per_request)
            print(f"----- request {i}, style {t['style']}, T={t['temperature']}\n" + user_prompt(t["topics"], t["style"], t["n"]))
        return
    os.makedirs(a.dir, exist_ok=True)
    raw_p, use_p, done_p = (os.path.join(a.dir, f) for f in ("raw.jsonl", "usage.jsonl", "done.txt"))
    done = set(int(x) for x in open(done_p).read().split()) if os.path.exists(done_p) else set()
    have = sum(1 for _ in open(raw_p, encoding="utf-8")) if os.path.exists(raw_p) else 0
    lock, stop = threading.Lock(), threading.Event()
    st = {"next": 0, "lines": have, "req": 0}
    print(f"resume: {len(done)} requests done, {have} lines; target {a.sentences} lines; workers {a.workers}")
    out, use, dn = open(raw_p, "a", encoding="utf-8"), open(use_p, "a", encoding="utf-8"), open(done_p, "a")

    def worker(wid):
        rng = random.Random(a.seed + 7919 * wid)
        while not stop.is_set():
            with lock:
                while st["next"] in done:
                    st["next"] += 1
                if st["lines"] >= a.sentences:
                    stop.set(); return
                i = st["next"]; st["next"] += 1
            t = task(i, topics, a.seed, a.per_request)
            msgs = [{"role": "system", "content": sysmsg},
                    {"role": "user", "content": user_prompt(t["topics"], t["style"], t["n"])}]
            resp = None
            for attempt in range(7):
                try:
                    resp = call_api(key, a.model, msgs, t["temperature"]); break
                except urllib.error.HTTPError as e:
                    if e.code in (400, 401, 402, 403, 422):        # bad request / key / balance: stop
                        print(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")
                        stop.set(); return
                    wait = min(300, 5 * 2 ** attempt) * rng.uniform(0.5, 1.5)
                    print(f"HTTP {e.code}, retry {attempt + 1} in {wait:.0f} s")
                except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                    wait = min(300, 5 * 2 ** attempt) * rng.uniform(0.5, 1.5)
                    print(f"network error {e}, retry {attempt + 1} in {wait:.0f} s")
                time.sleep(wait)
            if resp is None:
                print("giving up on request", i); continue
            ch = resp["choices"][0]
            lines = parse_lines(ch["message"].get("content") or "")
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            with lock:
                for x in lines:
                    out.write(json.dumps({"text": x, "platform": PLATFORM, "model": resp.get("model", a.model),
                                          "req": i, "style": t["style"], "topics": t["topics"],
                                          "temperature": t["temperature"], "time": ts}, ensure_ascii=False) + "\n")
                use.write(json.dumps({"req": i, "lines": len(lines), "finish": ch.get("finish_reason"),
                                      "usage": resp.get("usage", {})}) + "\n")
                dn.write(f"{i}\n")
                out.flush(); use.flush(); dn.flush()
                st["lines"] += len(lines); st["req"] += 1
                print(f"[{st['req']}] req {i} {t['style']:8s} -> {len(lines):2d} lines (total {st['lines']})"
                      + ("" if lines else f"  finish={ch.get('finish_reason')}"))
            time.sleep(rng.uniform(*a.delay))

    ths = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(a.workers)]
    for th in ths: th.start()
    try:
        for th in ths:
            while th.is_alive(): th.join(0.5)
    except KeyboardInterrupt:
        print("stopping after the requests in flight ..."); stop.set()
        for th in ths: th.join()
    out.close(); use.close(); dn.close()
    print("finished:", st["req"], "requests this run,", st["lines"], "lines in", raw_p)


# ---------------------------------------------------------------- stats
def usable(s, cs):
    return MINLEN <= len(s) <= MAXLEN and s[-1] in END and all(c in cs for c in s) \
        and not re.search(r"[，。！？、]{2}", s) and not netease_lint._pat.search(s)


def stats(a):
    rows = [json.loads(l) for l in open(os.path.join(a.dir, "raw.jsonl"), encoding="utf-8")]
    wiki = set(load_vocab())
    freq = {}
    for r in rows:
        for c in r["text"]:
            if "一" <= c <= "鿿": freq[c] = freq.get(c, 0) + 1
    own = set(PUNCT) | set(sorted(freq, key=lambda c: -freq[c])[:1024 - 4 - len(PUNCT)])
    tot = max(1, sum(freq.values()))
    print(f"distinct characters {len(freq)}; share of characters covered: by the wiki vocabulary "
          f"{sum(n for c, n in freq.items() if c in wiki) / tot:.1%}, by this corpus' own top {len(own) - len(PUNCT)} "
          f"{sum(n for c, n in freq.items() if c in own) / tot:.1%}")
    cs = own if a.own_vocab else wiki
    print("usable rate below is measured against", "this corpus' own top characters" if a.own_vocab else "the wiki vocabulary")
    ok, seen, why = [], set(), {"length": 0, "end": 0, "oov": 0, "punct": 0, "dup": 0}
    for r in rows:
        s = r["text"]
        if not usable(s, cs):
            if not MINLEN <= len(s) <= MAXLEN: why["length"] += 1
            elif s[-1] not in END: why["end"] += 1
            elif not all(c in cs for c in s): why["oov"] += 1
            else: why["punct"] += 1
            continue
        if s in seen:
            why["dup"] += 1; continue
        seen.add(s); ok.append(s)
    u = [json.loads(l)["usage"] for l in open(os.path.join(a.dir, "usage.jsonl"))]
    hit = sum(x.get("prompt_cache_hit_tokens", 0) for x in u)
    miss = sum(x.get("prompt_cache_miss_tokens", x.get("prompt_tokens", 0)) for x in u)
    outt = sum(x.get("completion_tokens", 0) for x in u)
    cost = (hit * a.price[0] + miss * a.price[1] + outt * a.price[2]) / 1e6
    n = max(1, len(rows))
    print(f"requests {len(u)}, raw lines {len(rows)}, usable unique {len(ok)} ({len(ok) / n:.1%}); rejected {why}")
    hist = {}
    for s in ok:
        k = f"{min(len(s) // 6 * 6, 30)}+"; hist[k] = hist.get(k, 0) + 1
    print("usable length histogram:", dict(sorted(hist.items(), key=lambda x: int(x[0][:-1]))))
    print(f"mean length {sum(map(len, ok)) / max(1, len(ok)):.1f} chars; styles "
          + str({k: sum(1 for r in rows if r['style'] == k) for k in STYLES}))
    print(f"tokens: input cache-hit {hit:,}  cache-miss {miss:,}  output {outt:,}")
    print(f"estimated cost ${cost:.3f} (prices per 1M: hit {a.price[0]}, miss {a.price[1]}, out {a.price[2]} USD; "
          f"the console bill is the truth) -> ${cost / max(1, len(ok)) * 1e6:.1f} per 1M usable sentences")
    oov = {}
    for r in rows:
        for c in r["text"]:
            if c not in cs and c not in PUNCT: oov[c] = oov.get(c, 0) + 1
    print("most frequent characters outside the vocabulary:", sorted(oov.items(), key=lambda x: -x[1])[:20])
    rng = random.Random(0)
    for s in rng.sample(ok, min(15, len(ok))):
        print("   ", s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make-topics", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--dir", default=OUT_DIR, help="output folder (raw.jsonl, usage.jsonl, done.txt)")
    ap.add_argument("--vocab-prompt", action="store_true", help="round-1 mode: put the wiki character list in the prompt")
    ap.add_argument("--own-vocab", action="store_true", help="--stats: judge characters against this corpus' own top 1020")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--sentences", type=int, default=10000, help="stop when raw.jsonl has this many lines")
    ap.add_argument("--per-request", type=int, default=50)
    ap.add_argument("--workers", type=int, default=2, help="requests in flight at once")
    ap.add_argument("--delay", type=float, nargs=2, default=[1.0, 3.0], metavar=("MIN", "MAX"))
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--price", type=float, nargs=3, default=[0.014, 0.14, 0.28], metavar=("HIT", "MISS", "OUT"),
                    help="USD per 1M tokens for the cost estimate in --stats")
    a = ap.parse_args()
    if a.make_topics: make_topics()
    elif a.stats: stats(a)
    else: run(a)


if __name__ == "__main__":
    main()
