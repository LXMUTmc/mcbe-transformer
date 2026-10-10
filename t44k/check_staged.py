"""Offline check of a STAGED build (any model size): compile with Compiler(staged=True), pack the
forward into stage chains, run the command TEXT chain by chain in the strict VM and compare every
tap of the forward with the integer spec. Also prints the stage table and the static budget.

  python -m t44k.check_staged CONFIG [n_prompts] [--stats] [--budget N]
  e.g.  python -m t44k.check_staged t1m 2
"""
import sys, os, json, time
import numpy as np
from t44k import spec as S
from t44k import layout
from t44k.compiler import Compiler, flatten, OBJ, nK, nV, nS, nW, nL
from t44k.check_offline import ChainVM


TRAINED = {"t1m": ("t44k/model_t1m/q.json", "data/wiki/vocab.json")}
# decoding rule of the trained model (chosen offline with t44k.eval_decode): presence penalty in
# integer logit units (2x the median top-1/top-2 gap) and no punctuation right after punctuation
DECODE = {"t1m": {"penalty": 354, "punct": "，。！？、"}}
# trained T1M variants: q.json, vocabulary, decoding penalty (2x median top-1/top-2 gap,
# t44k.eval_decode / t44k.compare_models). --model NAME selects one (default: wiki)
MODELS = {"wiki": ("t1m", "t44k/model_t1m/q.json", "data/wiki/vocab.json", 354),
          "kd":   ("t1m", "t44k/model_t1m_kd/q.json", "data/mix/vocab.json", 286),
          "tc":   ("t1m", "t44k/model_t1m_tc/q.json", "data/mix/vocab.json", 252),
          "c64":  ("t1m64", "t44k/model_t1m64/q.json", "data/mix64/vocab.json", None),
          "t4m":  ("t4m", "t44k/model_t4m/q.json", "data/mix64/vocab.json", None),
          "t4m2": ("t4m", "t44k/model_t4m_v2/q.json", "data/mix64b/vocab.json", None),
          # dialogue models (joint vocabulary; prompts are dialogue turns ending with <EOS>)
          "d7e":  ("t7m", "t44k/model_d7m_e/q.json", "data/dialog64d/vocab.json", None),
          "d10e": ("t10m", "t44k/model_d10m_e/q.json", "data/dialog64d/vocab.json", None),
          "d10f": ("t10m", "t44k/model_d10m_f/q.json", "data/dialog64d/vocab.json", None)}   # 160K steps + mined LCCC QA
DIALOG = {"d7e", "d10e", "d10f"}
DIALOG_MODE = False     # set by use_model: turn-local presence penalty, dialogue prompts


USE_CSE = False
USE_PACK = False        # --pack: embedding / position tables packed 3 per score (Compiler(pack=True))


def model_dir(path):
    """a self-contained model directory (released weights, or t44k.pipeline output): q.json, vocab.json
    and model.json {"penalty": int, "dialog": bool}. Returns a MODELS-style entry and the dialog flag."""
    q = json.load(open(os.path.join(path, "q.json")))
    cfg = next(k for k, c in S.CONFIGS.items() if c == q["config"])
    m = json.load(open(os.path.join(path, "model.json"), encoding="utf-8")) if os.path.exists(os.path.join(path, "model.json")) else {}
    return (cfg, os.path.join(path, "q.json"), os.path.join(path, "vocab.json"), m.get("penalty")), bool(m.get("dialog"))


def use_model(name):
    """name: a MODELS key, or a model directory (see model_dir)"""
    global DIALOG_MODE
    if os.path.isdir(name):
        (cfg, q, v, pen), DIALOG_MODE = model_dir(name)
    else:
        cfg, q, v, pen = MODELS[name]
        DIALOG_MODE = name in DIALOG
    if pen is None:                                  # 2x the median top-1/top-2 gap, measured on real prompts
        pen = measure_penalty(cfg, q, v)
        print(f"  decoding penalty for {name}: {pen}")
    TRAINED[cfg] = (q, v)
    DECODE[cfg] = {"penalty": pen, "punct": DECODE["t1m"]["punct"], "turn_reset": DIALOG_MODE}


def measure_penalty(cfg, qpath, vpath, n=40):
    S.configure(cfg)
    q = json.load(open(qpath))
    vocab = json.load(open(vpath, encoding="utf-8"))["vocab"]
    idx = {c: i for i, c in enumerate(vocab)}
    im, gaps = S.IntModel(q), []
    val = [l.strip() for l in open(os.path.join(os.path.dirname(vpath), "val.txt"), encoding="utf-8")]
    for s in [s for s in val if len(s) > 4 and all(c in idx for c in s[:2])][:n]:
        cache = im.new_cache()
        for pos, t in enumerate([S.BOS] + [idx[c] for c in s[:2]]):
            _, lg = im.step(t, pos, cache)
            v = sorted(int(lg[a]) for a in q["allowed"]); gaps.append(v[-1] - v[-2])
    gaps.sort()
    return 2 * (gaps[len(gaps) // 2] or 1)


def model_arg(argv):
    """pops '--model NAME' from argv and selects that model; '--cse' turns on matmul CSE (t44k/cse.py)"""
    global USE_CSE, USE_PACK
    if "--cse" in argv:
        USE_CSE = True; argv.remove("--cse")
    if "--pack" in argv:
        USE_PACK = True; argv.remove("--pack")
    if "--model" in argv:
        i = argv.index("--model"); use_model(argv[i + 1]); del argv[i:i + 2]


def build(config, seed=0, n_cal=16, budget=60000, trained=False, greedy=False):
    """trained=False: random validation weights; True: the trained model of this size with its
    real vocabulary (TRAINED) and its decoding rule (DECODE) unless greedy"""
    S.configure(config)
    if trained:
        qpath, vpath = TRAINED[config]
        q = json.load(open(qpath))
        vocab = json.load(open(vpath, encoding="utf-8"))["vocab"]
        assert q["config"] == S.CONFIGS[config], "model was trained for another size"
        if config in DECODE and not greedy:
            d = DECODE[config]
            idx = {ch: i for i, ch in enumerate(vocab)}
            q["decode"] = {"penalty": d["penalty"],
                           "exempt": sorted({idx[ch] for ch in d["punct"] if ch in idx} | {S.EOS}),
                           "turn_reset": d.get("turn_reset", False)}
        comp = Compiler(q, vocab, staged=True, cse=USE_CSE, pack=USE_PACK)
    else:
        q = S.random_model(seed)
        S.calibrate(q, S.random_prompts(n_cal, seed=seed + 1))
        comp = Compiler(q, staged=True, cse=USE_CSE, pack=USE_PACK)
    return q, comp, comp.stages(budget)


# test prompts when no validation text sits next to the vocabulary (released weights): everyday
# dialogue turns ('|' separates speakers) and plain sentence starts
FALLBACK_DIALOG = ["你好|你好呀", "吃饭了吗|还没呢|那快去吃吧", "我今天好累|早点休息吧", "周末你干嘛|在家睡觉|好羡慕你",
                   "明天要下雨|记得带伞|好的谢谢", "你喜欢猫吗|喜欢啊|我也是|你家里养了吗|没有养|为什么不养呢"]
FALLBACK_TEXT = ["今天天气很好。", "我们一起去公园散步。", "小猫在院子里睡觉。", "他每天早上都去跑步。"]


def real_prompts(n, config="t1m", n_chars=2):
    """the first n_chars characters of validation lines (no <UNK>), as token ids"""
    vocab = json.load(open(TRAINED[config][1], encoding="utf-8"))["vocab"]
    idx = {c: i for i, c in enumerate(vocab)}
    vp = os.path.join(os.path.dirname(TRAINED[config][1]), "val.txt")
    if os.path.exists(vp):
        val = [l.strip() for l in open(vp, encoding="utf-8")]
    else:
        val = [s for s in (FALLBACK_DIALOG if DIALOG_MODE else FALLBACK_TEXT) if all(c in idx or c == "|" for c in s)]
    out = []
    if DIALOG_MODE:
        # dialogue prompts: one turn + <EOS>, then (alternating) a 3-turn history + <EOS>; the reply
        # follows. n_chars > 2 asks for long histories (several turns, up to n_chars characters)
        for k, s in enumerate(v for v in val if "□" not in v):
            turns = s.split("|")
            want = 1 if (n_chars <= 2 and k % 2 == 0) else len(turns)
            take = []
            for t in turns[:want]:
                if len("|".join(take + [t])) > max(n_chars, 40):
                    break
                take.append(t)
            if not take or (want > 1 and len(take) < 2):
                continue
            out.append([S.BOS] + [S.EOS if c == "|" else idx[c] for c in "|".join(take)] + [S.EOS])
            if len(out) == n:
                break
        return out
    for s in val:
        p = s[:n_chars]
        if len(p) == n_chars and all(c in idx and idx[c] >= 4 for c in p):
            out.append([S.BOS] + [idx[c] for c in p])
        if len(out) == n:
            break
    return out


def compare_all(vm, tr, pos, mism, rm=None):
    g = (lambda n: vm.s.get(rm.get(n, (n, OBJ)))) if rm else (lambda n: vm.s.get((n, OBJ)))

    def chk(tap, names, exp):
        bad = sum(1 for n, e in zip(names, np.ravel(exp)) if g(n) != int(e))
        if bad:
            mism[tap] = mism.get(tap, 0) + bad
    chk("xe", [f"xe{i}" for i in range(S.D)], tr["xe"])
    for l in range(S.L):
        for nm in ("na", "q", "k", "v", "qs", "ks", "a", "xa", "nb", "f", "xf"):
            chk(f"{nm}{l}", [nL(nm, l, i) for i in range(S.F if nm == "f" else S.D)], tr[f"{nm}{l}"])
        chk(f"K{l}[pos]", [nK(l, pos, i) for i in range(S.D)], tr[f"ks{l}"])
        chk(f"V{l}[pos]", [nV(l, pos, i) for i in range(S.D)], tr[f"v{l}"])
        chk(f"s{l}", [nS(l, h, j) for h in range(S.H) for j in range(S.T)], tr[f"s{l}"])
        chk(f"w{l}", [nW(l, h, j) for h in range(S.H) for j in range(S.T)], tr[f"w{l}"])
    chk("nf", [f"nf{i}" for i in range(S.D)], tr["nf"])
    chk("best", ["bt", "blg"], [tr["best"], tr["bestLog"]])


def main(config, n_prompts=2, stats_only=False, budget=60000, trained=False, regmap=False, n_chars=2):
    t0 = time.time()
    q, comp, stages = build(config, budget=budget, trained=trained)
    rm = None
    if regmap:      # few holder names x many objectives (t44k/regmap.py); same integers
        from t44k import regmap as RM
        allc = [c.lstrip("?") for _, s in stages for c in flatten(s)] + comp.init_program() + comp.start_program([S.BOS, 5])
        rm = RM.build_map(RM.names_in(allc))
        stages = [(nm, [(sn, [RM.remap_cmd(c, rm) for c in cmds]) for sn, cmds in secs]) for nm, secs in stages]
        print(f"  regmap: {len(rm)} registers -> {RM.H} holder names x {len(RM.objectives(rm))} objectives")
    lens = [len(flatten(secs)) for _, secs in stages]
    try:
        plan = layout.build_plan_staged(stages)
    except AssertionError:      # too big for 64-wide columns in 10 ticking areas: count on the 192-wide footprint
        layout.set_footprint(192)
        plan = layout.build_plan_staged(stages, y_lo=-60, y_hi=318, max_chain=2_000_000)
    print(f"{config}: params {S.nominal_params():,}  stages {len(stages)}  commands/stage {lens}")
    print(f"  static command blocks {len(plan['blocks']):,} in {plan['columns']} column(s); "
          f"RAM at ~2.5 KB/block ~{len(plan['blocks']) * 2.5 / 1024 / 1024:.2f} GB  (built in {time.time() - t0:.0f} s)")
    if stats_only:
        return True
    prompts = real_prompts(n_prompts, config, n_chars) if trained else S.random_prompts(n_prompts, seed=7, max_len=3)
    ok = True
    for div_mode in ("trunc", "floor"):
        mism, faults, seqbad, fw, eff = {}, 0, 0, 0, 0
        for p in prompts:
            trace = []
            exp = S.IntModel(q).generate(p, trace=trace)
            vm = ChainVM(div_mode)
            if rm:
                vm.run_all(RM.init_objectives(rm) + [RM.remap_cmd(c, rm) for c in comp.init_program()])
                vm.run_all([RM.remap_cmd(c, rm) for c in comp.start_program(p)])
            else:
                vm.run_all(comp.init_program())
                vm.run_all(comp.start_program(p))
            step = 0
            while vm.s.get(("run", OBJ)) == 1:
                pos = vm.s.get(("pos", OBJ))
                e0 = vm.effective
                for k, (_, secs) in enumerate(stages):
                    vm.run(f"scoreboard players set stg {OBJ} {k}")         # the relay's job
                    vm.run_chain(flatten(secs))
                eff += vm.effective - e0
                if step >= len(trace):
                    mism["extra forward"] = mism.get("extra forward", 0) + 1
                    print("   commands ran a forward the spec did not; taps so far:", mism)
                    break
                compare_all(vm, trace[step], pos, mism, rm)
                step += 1
            got = [x for x in (vm.s.get((f"out{i}", OBJ)) for i in range(S.T)) if x != -1]
            seqbad += got != exp
            faults += len(vm.faults)
            fw += step
        print(f"  [{div_mode}] prompts {len(prompts)}, forwards {fw}, tap mismatches {sum(mism.values())} {mism or ''}, "
              f"sequence mismatches {seqbad}, VM faults {faults}, effective commands/forward {eff // max(1, fw):,}")
        ok &= not mism and not faults and not seqbad
    print("STAGED OFFLINE CHECK", "PASS" if ok else "FAIL", f"({time.time() - t0:.0f} s)")
    return ok


if __name__ == "__main__":
    a = sys.argv[1:]
    model_arg(a)
    budget = 60000
    if "--budget" in a:
        i = a.index("--budget"); budget = int(a[i + 1]); del a[i:i + 2]
    nch = 2
    if "--chars" in a:
        i = a.index("--chars"); nch = int(a[i + 1]); del a[i:i + 2]
    st, tr, rmf = "--stats" in a, "--trained" in a, "--regmap" in a
    a = [x for x in a if not x.startswith("--")]
    sys.exit(0 if main(a[0] if a else "t1m", int(a[1]) if len(a) > 1 else 2, st, budget, tr, rmf, nch) else 1)
