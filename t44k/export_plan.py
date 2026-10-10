"""Export a T44K build for the in-game builder (mod-debug-bridge, t44k/ingame.py):
  plan.json      every command block (+ ticking area, power spot(s), start/stop commands)
  programs.json  init program, per-prompt start programs, expected outputs and final-state snapshot,
                 and every score-holder name (probed in-game before placing)

  python -m t44k.export_plan OUT_DIR [x0] [seed] [layout: single|single_impulse|groups]
"""
import sys, os, json, re
import netease_lint
from t44k import spec as S
from t44k import layout
from t44k.check_offline import build
from t44k.compiler import snap


def main(out_dir, x0=16384, seed=0, which="single", n_prompts=24, model=None):
    layout.set_origin(x0)
    q, comp, _ = build(seed, model=model)
    if which == "groups":
        plan = layout.build_plan(comp.groups())
    else:
        plan = layout.build_plan_single(comp.groups(single=True), "impulse" if which == "single_impulse" else "repeat_rs")
    netease_lint.check([b["cmd"] for b in plan["blocks"]], "t44k plan")
    tests = []
    prompts = S.random_prompts(n_prompts, seed=seed + 11)
    if model and comp.vocab and os.path.exists("data/val.txt"):     # trained: real prompts (2 chars of val sentences)
        idx = {c: i for i, c in enumerate(comp.vocab)}
        val = [l.strip() for l in open("data/val.txt", encoding="utf-8") if l.strip()]
        prompts = [[S.BOS] + [idx[c] for c in s[:2]] for s in val[:n_prompts]]
    for p in prompts:
        im = S.IntModel(q)
        tr = []
        out = im.generate(p, trace=tr)
        last = tr[-1]
        final = {snap(f"xf{S.L-1}x{i}"): int(last[f"xf{S.L-1}"][i]) for i in range(S.D)}
        final.update({snap("bt"): int(last["best"]), snap("blg"): int(last["bestLog"])})
        tests.append({"prompt": p, "start": comp.start_program(p) + plan["start_extra"], "out": out, "final": final})
    init = comp.init_program()
    netease_lint.check(init, "t44k init")
    names = set()
    for c in [b["cmd"] for b in plan["blocks"]] + init + [x for t in tests for x in t["start"]]:
        names.update(re.findall(r"(\S+) (?:t4|t4c)(?= |$)", c))
    names = sorted(n for n in names if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", n))
    netease_lint.check_names(names, "t44k names")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(plan, open(os.path.join(out_dir, "plan.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump({"init": init, "tests": tests, "model_hash": S.qhash(q), "seed": seed, "names": names,
               "layout": plan["mode"]},
              open(os.path.join(out_dir, "programs.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("layout", plan["mode"], "blocks", len(plan["blocks"]), "chains", plan["chain_lengths"], "power", plan["power"],
          "start_extra", plan["start_extra"], "names", len(names), "model", S.qhash(q))


if __name__ == "__main__":
    a = sys.argv[1:]
    model = "trained" if "--trained" in a else None
    a = [x for x in a if not x.startswith("--")]
    main(a[0], int(a[1]) if len(a) > 1 else 16384, int(a[2]) if len(a) > 2 else 0, a[3] if len(a) > 3 else "single", model=model)
