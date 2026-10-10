"""A training project: any mix of corpora -> trained model -> Minecraft packs and world. This is what
the desktop app (app/mct_studio.py) runs; it can also be used from the command line.

  python -m t44k.project run projects/NAME/project.json      every stage, skipping finished ones
  python -m t44k.project pause projects/NAME/project.json    ask the training to stop at its next checkpoint

project.json:
  {"name": "mymodel",                 lowercase letters / digits / _ (file names, pack tag)
   "title": "我的模型",               shown in the game
   "config": "t4m",                    model size (t44k/spec.py CONFIGS: mini, t1m64, t4m, t7m, t10m)
   "steps": 60000, "batch": 256, "zero_frac": 0.7, "compile": true,
   "vocab": "released",                "released" (data/mixd/vocab.json: the released model's 1024
                                       characters) or "own" (built from the selected corpora)
   "sources": [                        any of, each with a "weight" (shares of every batch):
     {"kind": "teacher", "dir": "data/teacher2"}           DeepSeek sentences (data/gen_teacher.py)
     {"kind": "wiki", "file": "data/wiki_raw.json"}        Chinese Wikipedia
     {"kind": "lccc", "dir": "data/lccc", "dedull": true}  LCCC conversations (research use only)
     {"kind": "text", "files": ["a.txt"], "dialog": false} your own text
   ],
   "check": false}                     command-level integer check before building (slow for T10M)
Folders: projects/NAME/{data, train, model, build}; results in projects/NAME/build/输出.
projects/NAME/status.json: {"stage", "stages", "state": running|paused|done|error, "message"}.
"""
import sys, os, json, subprocess, time, hashlib, random, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PY = sys.executable
MEM = 11
STAGES = ["语料", "训练", "模型", "生成"]


class Project:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.dir = os.path.dirname(self.path)
        self.p = json.load(open(self.path, encoding="utf-8"))
        self.name = self.p["name"]
        assert self.name.replace("_", "a").isalnum() and self.name == self.name.lower() and self.name[0].isalpha(), \
            "name: lowercase letters, digits, _ (starting with a letter)"

    def sub(self, *a):
        return os.path.join(self.dir, *a)

    def status(self, stage, state="running", message=""):
        json.dump({"stage": stage, "stages": STAGES, "state": state, "message": message, "time": time.time()},
                  open(self.sub("status.json"), "w", encoding="utf-8"), ensure_ascii=False)
        print(f"[{stage}] {state} {message}", flush=True)

    def run(self, args, log, mem=MEM):
        cmd = [PY, "-u", "-m", "t44k.capped", str(mem)] + [str(a) for a in args] if mem else [PY, "-u"] + [str(a) for a in args]
        with open(self.sub(log), "a", encoding="utf-8") as f:
            f.write("$ " + " ".join(cmd[1:]) + "\n"); f.flush()
            r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            tail = open(self.sub(log), encoding="utf-8", errors="replace").read().splitlines()[-8:]
            raise RuntimeError(f"{' '.join(str(a) for a in args[:2])} failed (exit {r.returncode}):\n" + "\n".join(tail))

    # ------------------------------------------------------------ corpora
    def vocab_path(self):
        return self.sub("data", "vocab.json")

    def make_vocab(self):
        if os.path.exists(self.vocab_path()):
            return
        os.makedirs(self.sub("data"), exist_ok=True)
        if self.p.get("vocab", "released") == "released":
            v = json.load(open(os.path.join(ROOT, "data", "mixd", "vocab.json"), encoding="utf-8"))["vocab"]
        else:
            v = own_vocab(self.p["sources"])
        json.dump({"vocab": v}, open(self.vocab_path(), "w", encoding="utf-8"), ensure_ascii=False)

    def corpus_dirs(self):
        """[(dir, weight)] -- one prepared corpus per source"""
        return [(self.sub("data", f"{i}_{s['kind']}"), s["weight"]) for i, s in enumerate(self.p["sources"])]

    def prepare(self):
        self.make_vocab()
        v = self.vocab_path()
        for (d, _), s in zip(self.corpus_dirs(), self.p["sources"]):
            if os.path.exists(os.path.join(d, "train.txt")):
                continue
            k = s["kind"]
            self.status("语料", message=f"准备 {k}")
            tmp = d + "_tmp"
            if k == "teacher":
                self.run(["data/prep_mix.py", "--teacher", s["dir"], "--no-wiki", "--vocab-file", v, "--maxlen", 63, "--out", tmp], "prepare.log")
            elif k == "wiki":
                self.run(["data/prep_mix.py", "--wiki-only", "--wiki-file", s["file"], "--vocab-file", v, "--maxlen", 63, "--out", tmp], "prepare.log")
            elif k == "lccc":
                raw = tmp + "_raw" if s.get("dedull", True) else tmp
                if not os.path.exists(os.path.join(raw, "train.txt")):
                    self.run(["data/prep_dialog.py", "--vocab", v, "--src", s["dir"], "--out", raw], "prepare.log")
                import shutil
                shutil.copy(v, os.path.join(raw, "vocab.json"))          # dedull_dialog copies it along
                if s.get("dedull", True):
                    self.run(["data/dedull_dialog.py", "--src", raw, "--out", tmp], "prepare.log")
            elif k == "text":
                self.run(["data/prep_text.py"] + s["files"] + ["--kind", "dialog" if s.get("dialog") else "text",
                                                                "--vocab", v, "--out", tmp], "prepare.log")
            else:
                raise ValueError(f"unknown source kind {k}")
            import shutil
            shutil.copy(v, os.path.join(tmp, "vocab.json"))
            os.replace(tmp, d)

    # ------------------------------------------------------------ training
    def train(self):
        out = self.sub("train")
        if os.path.exists(os.path.join(out, "q.json")):
            return
        mix = [f"{d}={w}" for d, w in self.corpus_dirs()]
        args = ["-m", "t44k.train", "--config", self.p["config"], "--steps", self.p["steps"], "--batch", self.p.get("batch", 256),
                "--zero-frac", self.p.get("zero_frac", 0.7), "--out", out, "--mix"] + mix + ["--vocab", self.vocab_path(), "--resume"]
        if self.p.get("compile", True):
            args.append("--compile")
        self.status("训练", message=f"{self.p['config']}，{self.p['steps']} 步")
        self.run(args, "train.log", mem=None)
        pr = json.load(open(os.path.join(out, "progress.json"), encoding="utf-8"))
        if pr["status"] == "paused":
            raise Paused(f"训练暂停在第 {pr['step']} 步")

    def pause(self):
        open(os.path.join(self.sub("train"), "STOP"), "w").write("stop")

    # ------------------------------------------------------------ model directory
    def model(self):
        md = self.sub("model")
        if os.path.exists(os.path.join(md, "model.json")):
            return
        import shutil
        from t44k import spec as S
        from t44k.m7 import char_freq
        os.makedirs(md, exist_ok=True)
        shutil.copy(os.path.join(self.sub("train"), "q.json"), md)
        shutil.copy(self.vocab_path(), md)
        vocab = json.load(open(self.vocab_path(), encoding="utf-8"))["vocab"]
        dirs = [d for d, _ in self.corpus_dirs()]
        json.dump(char_freq([os.path.join(d, "train.txt") for d in dirs], vocab),
                  open(os.path.join(md, "char_freq.json"), "w", encoding="utf-8"), ensure_ascii=False, sort_keys=True)
        dialog = any(has_turns(os.path.join(d, "train.txt")) for d in dirs)
        val = [os.path.join(d, "val.txt") for d in dirs if has_turns(os.path.join(d, "train.txt")) == dialog]
        # 200 validation lines next to the vocabulary: the build's test prompts and the keyboard's self-test
        # come from real text of this project (check_staged.real_prompts); local only, never released
        lines = [l for p in val for l in open(p, encoding="utf-8") if l.strip()]
        lines = ([l for l in lines if "□" not in l] + [l for l in lines if "□" in l])[:200]   # usable as prompts first
        open(os.path.join(md, "val.txt"), "w", encoding="utf-8").write("".join(lines))
        q = json.load(open(os.path.join(md, "q.json")))
        cfg = next(k for k, c in S.CONFIGS.items() if c == q["config"])
        S.configure(cfg)
        log = json.load(open(os.path.join(self.sub("train"), "train_log.json"), encoding="utf-8"))
        json.dump({"name": self.name, "title": self.p.get("title") or f"MCBE Transformer {self.name}", "config": cfg,
                   "params": S.nominal_params(), "dialog": dialog, "penalty": penalty(q, cfg, vocab, val[0]),
                   "training": log.get("args"), "quality": {"int_top1": log.get("int_val_top1")},
                   "sources": self.p["sources"]},
                  open(os.path.join(md, "model.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ------------------------------------------------------------ build
    def build(self):
        from t44k import pipeline
        pipeline.build(self.sub("model"), self.sub("build"), self.name, self.p.get("check", False), MEM)

    def run_all(self):
        try:
            self.status("语料"); self.prepare()
            self.status("训练"); self.train()
            self.status("模型"); self.model()
            self.status("生成", message="导出、键盘、行为包、存档（大模型要几十分钟）"); self.build()
            self.status("生成", "done", self.sub("build", "输出"))
        except Paused as e:
            self.status("训练", "paused", str(e))
        except BaseException as e:      # SystemExit from pipeline.run too
            st = json.load(open(self.sub("status.json"), encoding="utf-8"))
            self.status(st["stage"], "error", str(e)[-1500:])
            raise


class Paused(Exception):
    pass


def has_turns(path):
    with open(path, encoding="utf-8") as f:
        return any("|" in l for _, l in zip(range(2000), f))


def penalty(q, cfg, vocab, val_path, n=40):
    """decoding penalty = 2x the median top-1/top-2 integer logit gap on validation prompts (as
    check_staged.measure_penalty)"""
    from t44k import spec as S
    S.configure(cfg)
    idx = {c: i for i, c in enumerate(vocab)}
    im, gaps = S.IntModel(q), []
    lines = [l.strip().split("|")[0] for l in open(val_path, encoding="utf-8")]
    for s in [s for s in lines if len(s) > 4 and all(c in idx for c in s[:2])][:n]:
        cache = im.new_cache()
        for pos, t in enumerate([S.BOS] + [idx[c] for c in s[:2]]):
            _, lg = im.step(t, pos, cache)
            v = sorted(int(lg[a]) for a in q["allowed"]); gaps.append(v[-1] - v[-2])
    gaps.sort()
    return 2 * ((gaps[len(gaps) // 2] if gaps else 0) or 1)


def own_vocab(sources, per_source=3_000_000):
    """1024 characters: specials + ，。！？、 + the best characters by frequency, every source weighted equally"""
    import gzip
    sys.path.insert(0, os.path.join(ROOT, "data"))
    from prep_wiki import SPECIALS, PUNCT, clean, articles
    from prep_text import read_text
    cjk = lambda s: (c for c in s if "一" <= c <= "鿿")
    score = collections.Counter()
    for s in sources:
        f = collections.Counter()
        if s["kind"] == "teacher":
            for i, l in enumerate(open(os.path.join(s["dir"], "raw.jsonl"), encoding="utf-8")):
                try:
                    f.update(cjk(clean(json.loads(l)["text"])))
                except (json.JSONDecodeError, KeyError):
                    pass
                if sum(f.values()) > per_source: break
        elif s["kind"] == "wiki":
            for t in articles(s["file"], limit=20000):
                f.update(cjk(clean(t)))
                if sum(f.values()) > per_source: break
        elif s["kind"] == "lccc":
            with gzip.open(os.path.join(s["dir"], "lccc_base_train.jsonl.gz"), "rt", encoding="utf-8") as fh:
                for l in fh:
                    for t in json.loads(l):
                        f.update(cjk(t.replace(" ", "")))
                    if sum(f.values()) > per_source: break
        elif s["kind"] == "text":
            for p in s["files"]:
                f.update(cjk(clean(read_text(p))))
        n = sum(f.values()) or 1
        for c, x in f.items():
            score[c] += x / n
    chars = list(PUNCT) + [c for c, _ in score.most_common(1024 - len(SPECIALS) - len(PUNCT))]
    v = SPECIALS + chars
    return v + [f"<pad{i}>" for i in range(1024 - len(v))]


if __name__ == "__main__":
    cmd, path = sys.argv[1], sys.argv[2]
    pr = Project(path)
    if cmd == "run":
        pr.run_all()
    elif cmd == "pause":
        pr.pause()
