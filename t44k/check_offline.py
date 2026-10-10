"""Offline M1-M4 check: run the compiled command TEXT (vm.VM, strict) group by group and compare
every tap with the exact integer spec (t44k/spec.IntModel).

  python -m t44k.check_offline [n_prompts] [seed]
"""
import sys, json
import numpy as np
from vm import VM
from t44k import spec as S
from t44k.compiler import Compiler, flatten, stats, OBJ, nK, nV, nS, nW, nL


class ChainVM(VM):
    """VM + conditional chain blocks ('?cmd' runs only if the previous block succeeded)"""
    def run_chain(self, cmds):
        last = True
        for c in cmds:
            if c.startswith("?"):
                if not last:
                    self.executed += 1
                    last = False
                    continue
                c = c[1:]
            r = self.run(c)
            last = bool(r)
        return last


TRAINED = "t44k/model/q.json"
VOCAB = "data/vocab_t44k.json"


def build(seed=0, n_cal=60, model=None):
    """model=None: random validation model; model='trained' (or a path): the exported integer
    model from t44k/train.py, compiled with the real character vocabulary"""
    import os
    if model:
        path = TRAINED if model == "trained" else model
        q = json.load(open(path))
        vocab = json.load(open(VOCAB, encoding="utf-8"))["vocab"] if os.path.exists(VOCAB) else None
        comp = Compiler(q, vocab)
    else:
        q = S.random_model(seed)
        S.calibrate(q, S.random_prompts(n_cal, seed=seed + 1))
        comp = Compiler(q)
    groups = comp.groups()
    return q, comp, groups


def g(vm, n):
    return vm.s.get((n, OBJ))


def compare(vm, tr, pos, which, mism):
    """which: 'G0'|'G1'|'G2' -> compare the taps produced by that group"""
    def chk(tap, names, exp):
        bad = sum(1 for n, e in zip(names, np.ravel(exp)) if g(vm, n) != int(e))
        if bad:
            mism[tap] = mism.get(tap, 0) + bad
    if which == "G0":
        chk("xe", [f"xe{i}" for i in range(S.D)], tr["xe"])
    layers = {"G0": [0], "G1": [1], "G2": []}[which]
    for l in layers:
        chk(f"na{l}", [nL("na", l, i) for i in range(S.D)], tr[f"na{l}"])
        for nm in ("q", "k", "v", "qs", "ks"):
            chk(f"{nm}{l}", [nL(nm, l, i) for i in range(S.D)], tr[f"{nm}{l}"])
        chk(f"K{l}[pos]", [nK(l, pos, i) for i in range(S.D)], tr[f"ks{l}"])
        chk(f"V{l}[pos]", [nV(l, pos, i) for i in range(S.D)], tr[f"v{l}"])
        chk(f"s{l}", [nS(l, h, j) for h in range(S.H) for j in range(S.T)], tr[f"s{l}"])
        chk(f"w{l}", [nW(l, h, j) for h in range(S.H) for j in range(S.T)], tr[f"w{l}"])
        for nm in ("a", "xa", "nb", "f", "xf"):
            n = S.F if nm == "f" else S.D
            chk(f"{nm}{l}", [nL(nm, l, i) for i in range(n)], tr[f"{nm}{l}"])
    if which == "G2":
        chk("nf", [f"nf{i}" for i in range(S.D)], tr["nf"])
        chk("best", ["bt", "blg"], [tr["best"], tr["bestLog"]])


def run_prompt(comp, groups, q, prompt, div_mode, mism):
    im = S.IntModel(q)
    trace = []
    exp_out = im.generate(prompt, trace=trace)
    vm = ChainVM(div_mode)
    vm.run_all(comp.init_program())
    vm.run_all(comp.start_program(prompt))
    step, got_out = 0, []
    while g(vm, "run") == 1:
        pos = g(vm, "pos")
        tr = trace[step]
        for gi, (gname, secs) in enumerate(groups):
            vm.run(f"scoreboard players set grp {OBJ} {gi}")      # the controller's job
            vm.run_chain(flatten(secs))
            compare(vm, tr, pos, gname, mism)
        step += 1
        assert step <= S.T + 1, "runaway"
    got_out = [g(vm, f"out{p}") for p in range(S.T) if g(vm, f"out{p}") != -1]
    return exp_out, got_out, vm, step


def main(n_prompts=40, seed=0, model=None):
    q, comp, groups = build(seed, model=model)
    st = stats(groups)
    print("commands per tick group:", {k: v["commands"] for k, v in st.items()},
          "total", sum(v["commands"] for v in st.values()))
    for gname, v in st.items():
        print(f"  {gname}: {v['by_section']}  (conditional blocks {v['conditional']})")
    prompts = S.random_prompts(n_prompts, seed=seed + 7)
    ok = True
    for div_mode in ("trunc", "floor"):
        mism, faults, seqbad, steps = {}, 0, 0, 0
        executed = effective = 0
        for p in prompts:
            exp_out, got_out, vm, n = run_prompt(comp, groups, q, p, div_mode, mism)
            faults += len(vm.faults)
            seqbad += exp_out != got_out
            steps += n
            executed += vm.executed; effective += vm.effective
        print(f"[{div_mode}] prompts {len(prompts)}, forwards {steps}, tap mismatches {sum(mism.values())} {mism or ''}, "
              f"sequence mismatches {seqbad}, VM faults {faults}, avg commands looked at / forward {executed // steps}, "
              f"effective {effective // steps}")
        ok &= not mism and not faults and not seqbad
    print("OFFLINE CHECK", "PASS" if ok else "FAIL")
    return ok, q, comp


if __name__ == "__main__":
    a = sys.argv[1:]
    model = "trained" if "--trained" in a else None
    a = [x for x in a if not x.startswith("--")]
    ok = main(int(a[0]) if a else 40, int(a[1]) if len(a) > 1 else 0, model)[0]
    sys.exit(0 if ok else 1)
