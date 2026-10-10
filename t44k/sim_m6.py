"""Offline tick simulation of the M6 world: init button -> type characters -> 生成 -> 自检.

  python -m t44k.sim_m6
"""
import sys
from t44k import spec as S
from t44k.compiler import OBJ
from t44k.check_offline import build
from t44k.m6 import build_m6
from t44k.sim import World


def run(mode, lag, text):
    m = build_m6()
    q, comp, _ = build(model="trained")
    keys = {k["label"]: k["cmds"] for k in m["keys"]}
    w = World({"blocks": m["blocks"]}, mode, lag, seed=lag)

    def press(label, ticks=2):
        w.run_all(keys[label])
        for _ in range(ticks):
            w.tick()

    # in a fresh world the always-on poller fails silently until 初始化 creates the objectives;
    # the strict VM would assert instead, so the objectives exist from the start here
    w.run_all(["scoreboard objectives add t4 dummy", "scoreboard objectives add t4c dummy",
               "scoreboard players set tk t4 0"])
    press("初始化", 3)
    assert w.s.get(("plen", OBJ)) == 1, "init did not run"
    for ch in text:
        press(ch)
    press("生成", 1)
    n = 0
    while w.s.get(("run", OBJ)) == 1 and n < 60:
        w.tick(); n += 1
    for _ in range(4):
        w.tick()
    idx = {c: i for i, c in enumerate(comp.vocab)}
    exp = S.IntModel(q).generate([S.BOS] + [idx[c] for c in text])
    got = [x for x in (w.s.get((f"out{p}", OBJ)) for p in range(S.T)) if x != -1]
    gen_ok = got == exp
    n0 = len(w.out)
    press("自检", 1)
    n = 0
    while w.s.get(("run", OBJ)) == 1 and n < 60:
        w.tick(); n += 1
    for _ in range(4):
        w.tick()
    st_msgs = [l for l in w.out[n0:] if "自检" in l]
    st_ok = any("PASS" in l for l in st_msgs) and not any("FAIL" in l for l in st_msgs)
    word = "".join("<EOS>" if t == S.EOS else comp.vocab[t] for t in got)
    print(f"  {mode:9s} lag {lag}: typed {text} -> {word} (matches spec: {gen_ok}); self-test: {st_msgs[-2:] if st_msgs else 'none'}; faults {len(w.faults)} {w.faults[:3]}")
    return gen_ok and st_ok and not w.faults


if __name__ == "__main__":
    ok = all(run(m, lag, t) for m, lag, t in (("next", 0, "石镐"), ("next", 1, "熟牛"), ("next+dup", 2, "我挖")))
    print("M6 SIM", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
