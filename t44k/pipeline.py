"""One-command pipeline: train a model, or turn a model into an installable Minecraft behavior pack.

  python -m t44k.pipeline build MODEL_DIR OUT_DIR [--tag NAME] [--check] [--mem GB]
      MODEL_DIR: q.json + vocab.json + model.json (+ char_freq.json for the pinyin keyboard), e.g. the
      released models/d10f or the output of 'train'. Steps (each one a separate process with a memory
      cap, skipped when its output exists):
        check   (--check) command-level check against the integer spec: 2 dialogues + 1 long one,
                both division semantics; must say PASS. T10M: about 3 hours the first time (matrix CSE,
                cached in t44k/cse_cache), minutes afterwards
        export  OUT_DIR/export   single-chain build (blocks.txt, meta.json, programs.json)
        m7      OUT_DIR/m7       pinyin keyboard (m7.json)
        pack    OUT_DIR/pack     behavior pack: NAME.mcpack, NAME_netease.zip (see t44k/pack.py)
        verify  read every structure tile back and compare with the export

  python -m t44k.pipeline train RECIPE [--from STEP] [--dry-run] [--mem GB]
      RECIPE: d10f (the released T10M dialogue model). Prints / runs every data and training step
      in order; steps whose output exists are skipped. Downloads are NOT automatic: the steps that
      need a file you must fetch tell you where it goes (see docs/训练指南.md).

Every command is printed before it runs, so the pipeline doubles as documentation of the recipe.
"""
import sys, os, json, subprocess, shutil, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def run(args, mem, dry=False, log=None):
    cmd = [PY, "-u", "-m", "t44k.capped", str(mem)] + [str(a) for a in args] if mem else [PY, "-u"] + [str(a) for a in args]
    print("$ " + " ".join(cmd[1:]), flush=True)
    if dry:
        return
    t0 = time.time()
    if log:
        with open(log, "w", encoding="utf-8") as f:
            p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        tail = open(log, encoding="utf-8", errors="replace").read().splitlines()[-6:]
        print("\n".join("    " + l for l in tail))
    else:
        p = subprocess.run(cmd, cwd=ROOT)
    if p.returncode != 0:
        raise SystemExit(f"step failed (exit {p.returncode}){' -- see ' + log if log else ''}")
    print(f"    done in {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------- build
def build(model, out, tag=None, check=False, mem=11):
    model, out = os.path.abspath(model), os.path.abspath(out)
    m = json.load(open(os.path.join(model, "model.json"), encoding="utf-8"))
    q = json.load(open(os.path.join(model, "q.json")))
    sys.path.insert(0, ROOT)
    from t44k import spec as S
    cfg = next(k for k, c in S.CONFIGS.items() if c == q["config"])
    tag = tag or m.get("name") or "mct"
    os.makedirs(out, exist_ok=True)
    common = ["--trained", "--regmap", "--model", model, "--cse", "--pack"]
    print(f"model {model}: config {cfg}, dialog {bool(m.get('dialog'))}, decoding penalty {m.get('penalty')}")
    if check:
        run(["t44k.check_staged", cfg, 2] + common, mem, log=os.path.join(out, "check.log"))
        run(["t44k.check_staged", cfg, 1] + common + ["--chars", S.CONFIGS[cfg]["T"] - 8], mem, log=os.path.join(out, "check_long.log"))
        for f in ("check.log", "check_long.log"):
            assert "STAGED OFFLINE CHECK PASS" in open(os.path.join(out, f), encoding="utf-8").read(), f"{f}: not PASS"
    ex, m7, pk = os.path.join(out, "export"), os.path.join(out, "m7"), os.path.join(out, "pack")
    if not os.path.exists(os.path.join(ex, "programs.json")):
        run(["t44k.export_staged", ex, cfg, 32768, 4, "single"] + common + ["--max-chain", 4000000, "--y0", -60],
            mem, log=os.path.join(out, "export.log"))
    if not os.path.exists(os.path.join(m7, "m7.json")):
        run(["t44k.m7", ex, m7, "--model", model] + (["--dialog"] if m.get("dialog") else []), mem, log=os.path.join(out, "m7.log"))
    if not os.path.exists(os.path.join(pk, f"{tag}.mcpack")):
        run(["t44k.pack", ex, m7, pk, "--tag", tag, "--name", m.get("title", f"MCBE Transformer {tag}")], mem,
            log=os.path.join(out, "pack.log"))
    run(["t44k.pack", "--verify", pk, ex, m7], mem, log=os.path.join(out, "verify.log"))
    title = m.get("title", f"MCBE Transformer {tag}")
    pa = os.path.join(out, "pack_auto")
    if not os.path.exists(os.path.join(pa, f"{tag}_auto.mcpack")):
        run(["t44k.pack", ex, m7, pa, "--tag", tag, "--name", title + "（自动安装）", "--auto"], mem,
            log=os.path.join(out, "pack_auto.log"))
    res = os.path.join(out, "输出")
    os.makedirs(res, exist_ok=True)
    world = os.path.join(res, f"{tag}.mcworld")
    if not os.path.exists(world):
        run(["-m", "t44k.world_build", pa, world, "--name", title], None, log=os.path.join(out, "world.log"))
    for src, dst in ((os.path.join(pa, f"{tag}_auto_netease.zip"), f"{tag}_网易版组件_自动安装.zip"),
                     (os.path.join(pa, f"{tag}_auto.mcpack"), f"{tag}_国际版行为包_自动安装.mcpack"),
                     (os.path.join(pk, f"{tag}_netease.zip"), f"{tag}_网易版组件_手动安装.zip"),
                     (os.path.join(pk, f"{tag}.mcpack"), f"{tag}_国际版行为包_手动安装.mcpack")):
        shutil.copy(src, os.path.join(res, dst))
    open(os.path.join(res, "怎么用.txt"), "w", encoding="utf-8").write(USAGE.format(tag=tag))
    print(f"\nready: {res}")


USAGE = """网易版（MC Studio）：
  1. 导入 {tag}_网易版组件_自动安装.zip
  2. 新建超平坦世界、开启作弊，挂上这个组件，进入世界
  3. 什么都不用做：15 秒后自动安装（游戏会卡十几次），装好后键盘出现在你面前，并自动初始化
国际版：
  双击 {tag}.mcworld 导入，打开世界后同样会自动安装
  （或：双击 {tag}_国际版行为包_自动安装.mcpack，挂到新建的超平坦世界上）
手动安装版（_手动安装）：新世界里输入 /function mct/install，装好后面朝南输入 /function mct/keyboard，再按 初始化
使用：字母键打拼音，数字键选字，按 发送；看到屏幕下方提示变了再按下一个键；按 自检 可以逐整数核对；存档退出前先按 关机
"""


# ---------------------------------------------------------------- train recipes
def need(path, what, dry=False):
    if not os.path.exists(os.path.join(ROOT, path)):
        msg = f"missing {path}: {what} (docs/训练指南.md)"
        if dry:
            print("    (needs) " + msg)
        else:
            raise SystemExit(msg)


def need_key(dry):
    if not dry and not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("set the environment variable DEEPSEEK_API_KEY first (never commit or print it)")


def concat_dialog64f(dry):
    """dialog64f = dialog64e + the mined QA pairs twice (train); dialog64e val + QA val"""
    print("$ concat data/dialog64e + 2 x data/qa_lccc -> data/dialog64f", flush=True)
    if dry:
        return
    d = os.path.join(ROOT, "data")
    os.makedirs(os.path.join(d, "dialog64f"), exist_ok=True)
    for split, parts in (("train", ["dialog64e", "qa_lccc", "qa_lccc"]), ("val", ["dialog64e", "qa_lccc"])):
        with open(os.path.join(d, "dialog64f", f"{split}.txt"), "wb") as f:
            for p in parts:
                f.write(open(os.path.join(d, p, f"{split}.txt"), "rb").read())
    shutil.copy(os.path.join(d, "mixd", "vocab.json"), os.path.join(d, "dialog64f", "vocab.json"))
    json.dump({"dialogues": "data/dialog64e (LCCC, de-dulled)", "qa": "data/qa_lccc (mined LCCC pairs, x2 in train)"},
              open(os.path.join(d, "dialog64f", "stats.json"), "w"), indent=1)


def model_dir(train_out, dest, name, dialog, mem, dry):
    """train output (q.json) -> self-contained model directory for 'build'"""
    print(f"$ model directory {dest}", flush=True)
    if dry:
        return
    sys.path.insert(0, ROOT)
    from t44k.m7 import char_freq
    from t44k.check_staged import measure_penalty
    from t44k import spec as S
    os.makedirs(dest, exist_ok=True)
    shutil.copy(os.path.join(train_out, "q.json"), dest)
    shutil.copy(os.path.join(ROOT, "data", "mixd", "vocab.json"), dest)
    vocab = json.load(open(os.path.join(dest, "vocab.json"), encoding="utf-8"))["vocab"]
    json.dump(char_freq([os.path.join(ROOT, "data", "mix64d", "train.txt"), os.path.join(ROOT, "data", "dialog64e", "train.txt")], vocab),
              open(os.path.join(dest, "char_freq.json"), "w", encoding="utf-8"), ensure_ascii=False, sort_keys=True)
    q = json.load(open(os.path.join(dest, "q.json")))
    cfg = next(k for k, c in S.CONFIGS.items() if c == q["config"])
    pen = measure_penalty(cfg, os.path.join(dest, "q.json"), os.path.join(ROOT, "data", "dialog64d", "vocab.json"))
    log = json.load(open(os.path.join(train_out, "train_log.json"), encoding="utf-8"))
    json.dump({"name": name, "config": cfg, "params": S.nominal_params(), "dialog": dialog, "penalty": pen,
               "training": log.get("args"), "quality": {"int_top1": log.get("int_val_top1")},
               "licence": "research / non-commercial only when trained on LCCC; see DATA_AND_MODEL.md"},
              open(os.path.join(dest, "model.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def recipe_d10f(dry, start, mem):
    """the released T10M dialogue model (2026-10-06/08). GPU hours on an RTX 4070 SUPER."""
    S_ = [
        ("topics", "data/teacher/topics.txt",
         lambda: (need("data/wiki_raw.json", "Chinese Wikipedia (0xDing/wikipedia-cn-20230720-filtered, wikipedia-cn-20230720-filtered.json)", dry),
                  run(["data/gen_teacher.py", "--make-topics"], None, dry))),
        ("teacher", "data/teacher2/raw.jsonl",          # ~2.15M sentences, about 10 USD (DEEPSEEK_API_KEY)
         lambda: (need_key(dry),
                  run(["data/gen_teacher.py", "--dir", "data/teacher2", "--sentences", 2150000, "--workers", 4], None, dry))),
        ("vocab", "data/mixd/vocab.json",               # the released vocabulary is in the repository: keep it to reuse the weights
         lambda: (need("data/lccc/lccc_base_train.jsonl.gz", "LCCC base (huggingface.co/datasets/silver/lccc)", dry),
                  run(["data/build_joint_vocab.py"], None, dry))),
        ("mix", "data/mix64d/train.txt",
         lambda: run(["data/prep_mix.py", "--teacher", "data/teacher2", "--out", "data/mix64d", "--maxlen", 63,
                      "--vocab-file", "data/mixd/vocab.json"], mem, dry)),
        ("dialog", "data/dialog64d/train.txt",
         lambda: run(["data/prep_dialog.py", "--vocab", "data/mixd/vocab.json", "--out", "data/dialog64d"], mem, dry)),
        ("dedull", "data/dialog64e/train.txt",
         lambda: run(["data/dedull_dialog.py", "--src", "data/dialog64d", "--out", "data/dialog64e"], mem, dry)),
        ("judge", "t44k/ta_dialog/ta.pt",               # float judge for QA mining, ~30 min
         lambda: run(["-m", "t44k.train_ta", "--config", "t1m64", "--data", "data/mix64d", "--aux-frac", 0.15,
                      "--extra", "data/dialog64d", "--extra-frac", 0.5, "--steps", 30000, "--compile", "--out", "t44k/ta_dialog"], None, dry)),
        ("qa", "data/qa_lccc/train.txt",
         lambda: run(["data/mine_qa.py", "--keep", 0.6, "--out", "data/qa_lccc"], None, dry)),
        ("dialog64f", "data/dialog64f/train.txt", lambda: concat_dialog64f(dry)),
        ("train", "t44k/model_d10m_f/q.json",           # T10M, 160K steps x 256: about 9 hours
         lambda: run(["-m", "t44k.train", "--config", "t10m", "--data", "data/mix64d", "--aux-frac", 0.15,
                      "--extra", "data/dialog64f", "--extra-frac", 0.45, "--steps", 160000, "--zero-frac", 0.7,
                      "--compile", "--out", "t44k/model_d10m_f"], None, dry)),
        ("model", "models/d10f_retrained/model.json",
         lambda: model_dir(os.path.join(ROOT, "t44k", "model_d10m_f"), os.path.join(ROOT, "models", "d10f_retrained"),
                           "d10f", True, mem, dry)),
    ]
    names = [n for n, _, _ in S_]
    if start and start not in names:
        raise SystemExit(f"--from: one of {names}")
    go = not start
    for name, out, fn in S_:
        go = go or name == start
        if not go:
            continue
        if os.path.exists(os.path.join(ROOT, out)) and not dry:
            print(f"[{name}] skip: {out} exists")
            continue
        print(f"[{name}] -> {out}")
        fn()
    print("\nnext: python -m t44k.pipeline build models/d10f_retrained OUT_DIR --check")


RECIPES = {"d10f": recipe_d10f}


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda k, d: (a.pop(a.index(k) + 1), a.remove(k))[0] if k in a else d
    flag = lambda k: (a.remove(k), True)[1] if k in a else False
    mem = float(opt("--mem", 11))
    if not a or a[0] not in ("build", "train"):
        raise SystemExit(__doc__)
    if a[0] == "build":
        tag = opt("--tag", None)
        build(a[1], a[2], tag, flag("--check"), mem)
    else:
        start, dry = opt("--from", None), flag("--dry-run")
        RECIPES[a[1]](dry, start, mem)
