"""Export a STAGED build (any model size, random validation weights) for t44k/ingame_staged.py:
  blocks.txt     one command block per line: "x y z kind facing cond auto<TAB>command" (streamed
                 in-game, so a million blocks never sit in one JSON object)
  meta.json      ticking areas, power spots, start/stop commands, stage table
  programs.json  init program, per-prompt start programs, expected outputs + final-state snapshot,
                 every score-holder name (probed in-game first)

  python -m t44k.export_staged OUT_DIR CONFIG [x0] [n_prompts]
"""
import sys, os, json, re
import netease_lint
from t44k import spec as S
from t44k import layout
from t44k.check_staged import build, real_prompts
from t44k.compiler import snap, flatten


HEAD = "repeat_brk"         # single mode: chain head (layout.build_plan_single: repeat_brk | repeat_rs | impulse_wd)
MAX_CHAIN = 2_000_000       # single mode: the gamerule maxcommandchainlength must be >= the chain (--max-chain)


def main(out_dir, config, x0=32768, n_prompts=3, mode="staged", trained=False, budget=60000, regmap=False):
    """mode 'staged': one chain per stage, relayed tick by tick (build_plan_staged)
       mode 'single': the whole forward in ONE chain on a 192-wide footprint, repeating head
                      (needs /gamerule maxcommandchainlength raised; one forward per tick)
       mode 'sweep' : staged with big stages (budget commands each) on the 192-wide footprint,
                      chains may exceed 65,535 blocks (segment sweep of the single-tick knee)"""
    if mode in ("single", "sweep"):
        layout.set_footprint(192)
    layout.set_origin(x0)
    q, comp, stages = build(config, trained=trained, budget=budget)
    if mode == "sweep":
        plan = layout.build_plan_staged(stages, y_lo=-60, y_hi=318, max_chain=1_000_000)
    elif mode == "single":
        stages = comp.single_staged()
        plan = layout.build_plan_single(stages, HEAD, max_chain=MAX_CHAIN)   # gamerule maxcommandchainlength must match
        top = max(b["pos"][1] for b in plan["blocks"])
        assert top <= 300, f"the chain reaches y={top}: start lower (--y0 -60)"
        plan["tickingareas"] = [list(plan["tickingarea"])]
        plan["columns"] = 1
    else:
        plan = layout.build_plan_staged(stages)
    cmds = [b["cmd"] for b in plan["blocks"]]
    netease_lint.check(cmds, f"{config} plan")
    tests = []
    prompts = real_prompts(n_prompts, config) if trained else S.random_prompts(n_prompts, seed=11, max_len=3)
    if trained and S.T > 32 and len(prompts) > 1:      # one long prompt so the acceptance reaches the last positions
        prompts[-1] = real_prompts(1, config, S.T - 6)[0]
    word = lambda ids: "".join("<EOS>" if t == S.EOS else comp.vocab[t] for t in ids)
    for p in prompts:
        tr = []
        out = S.IntModel(q).generate(p, trace=tr)
        last = tr[-1]
        final = {snap(f"xf{S.L-1}x{i}"): int(last[f"xf{S.L-1}"][i]) for i in range(S.D)}
        final.update({snap("bt"): int(last["best"]), snap("blg"): int(last["bestLog"])})
        tests.append({"prompt": p, "start": comp.start_program(p) + plan["start_extra"], "out": out,
                      "final": final, "forwards": len(tr), "text": word(p[1:]) + "→" + word(out)})
    init = comp.init_program()
    if regmap:      # few holder names x many objectives (t44k/regmap.py): same integers, no per-tick blow-up
        from t44k import regmap as RM
        rm = RM.build_map(RM.names_in(cmds + init + [x for t in tests for x in t["start"]]))
        for b in plan["blocks"]:
            b["cmd"] = RM.remap_cmd(b["cmd"], rm)
        init = init[:2] + RM.init_objectives(rm) + [RM.remap_cmd(c, rm) for c in init[2:]]
        for t in tests:
            t["start"] = [RM.remap_cmd(c, rm) for c in t["start"]]
        cmds = [b["cmd"] for b in plan["blocks"]]
        netease_lint.check(cmds, f"{config} plan (regmap)")
        print(f"regmap: {len(rm)} registers -> {RM.H} holder names x {len(RM.objectives(rm))} objectives")
    netease_lint.check(init, f"{config} init")
    names = set()
    for c in cmds + init + [x for t in tests for x in t["start"]]:
        names.update(re.findall(r"(\S+) (?:t4|t4c|v\d+)(?= |$)", c))
    names = sorted(n for n in names if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", n))
    netease_lint.check_names(names, f"{config} names")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "blocks.txt"), "w", encoding="utf-8", newline="\n") as f:
        for b in plan["blocks"]:
            assert "\t" not in b["cmd"] and "\n" not in b["cmd"]
            f.write("%d %d %d %s %d %d %d\t%s\n" % (*b["pos"], b["kind"], b["facing"], b["cond"], b["auto"], b["cmd"]))
    meta = {k: plan[k] for k in ("mode", "tickingareas", "power", "chain_lengths", "columns", "start_extra", "stop_extra")}
    meta.update(config=config, x0=x0, T=S.T, n_blocks=len(plan["blocks"]), params=S.nominal_params(),
                stage_commands=[len(flatten(s)) for _, s in stages], model_hash=S.qhash(q))
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump({"init": init, "tests": tests, "names": names, "model_hash": S.qhash(q)},
              open(os.path.join(out_dir, "programs.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(config, "blocks", len(plan["blocks"]), "stages", len(stages), "columns", plan["columns"],
          "names", len(names), "init", len(init), "tests", [(t["prompt"], t["forwards"]) for t in tests], "model", S.qhash(q))


if __name__ == "__main__":
    a = sys.argv[1:]
    from t44k.check_staged import model_arg
    model_arg(a)
    tr, rmf = "--trained" in a, "--regmap" in a
    if "--head" in a:
        i = a.index("--head"); HEAD = a[i + 1]; del a[i:i + 2]
    if "--max-chain" in a:
        i = a.index("--max-chain"); MAX_CHAIN = int(a[i + 1]); del a[i:i + 2]
    if "--y0" in a:             # lowest layer of the chain (T10M: 257 layers do not fit above y=100)
        i = a.index("--y0"); layout.Y0 = int(a[i + 1]); del a[i:i + 2]
    budget = 60000
    if "--budget" in a:
        i = a.index("--budget"); budget = int(a[i + 1]); del a[i:i + 2]
    a = [x for x in a if not x.startswith("--")]
    main(a[0], a[1], int(a[2]) if len(a) > 2 else 32768, int(a[3]) if len(a) > 3 else 3, a[4] if len(a) > 4 else "staged", tr, budget, rmf)
