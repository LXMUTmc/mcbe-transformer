"""Tick-level world simulation of a T44K placement plan (t44k/layout.py).

Modelled Bedrock behaviour (measured in-game where noted):
  - conditional chain blocks test the command block GEOMETRICALLY BEHIND them (measured)
  - impulse heads: 'immediate' (inside the /setblock), 'next' tick (what NetEase does, measured),
    'next+dup' (fires once more the tick after: NetEase-style extra execution)
  - repeating heads that need redstone keep running LAG ticks after their power is removed
    (NetEase, measured in v0.1.1); heads/impulses run in a random order within a tick

  python -m t44k.sim [n_prompts] [layout: groups|single|single_impulse|all]
"""
import sys, random
from t44k import spec as S
from t44k.compiler import OBJ, snap
from t44k import layout
from t44k.check_offline import ChainVM, build

DIRS = {0: (0, -1, 0), 1: (0, 1, 0), 2: (0, 0, -1), 3: (0, 0, 1), 4: (-1, 0, 0), 5: (1, 0, 0)}


class World(ChainVM):
    def __init__(self, plan, mode="next", lag=0, seed=0):
        super().__init__("trunc")
        self.blk = {tuple(b["pos"]): dict(b) for b in plan["blocks"]}
        self.redstone = set()
        self.mode, self.lag = mode, lag
        self.pending = []
        self.rng = random.Random(seed)
        self.hist = {}
        self.forwards_run = 0
        self.firing = set()
        self.fires = {}
        self.full = {}

    def powered(self, p):
        return any(tuple(p[j] + d[j] for j in range(3)) in self.redstone for d in DIRS.values())

    def run(self, cmd):
        t = cmd.split(" ")
        if t[0] == "setblock":
            self.executed += 1; self.effective += 1
            p = tuple(int(v) for v in t[1:4])
            if t[4] == "redstone_block":
                was = p in self.redstone
                self.redstone.add(p)
                if not was:
                    for d in DIRS.values():
                        n = tuple(p[j] + d[j] for j in range(3))
                        b = self.blk.get(n)
                        if b and b["kind"] == "impulse":
                            if n in self.firing and self.mode.endswith("noreentry"):
                                continue      # NetEase (measured 2026-10-02): a chain re-powering its own head is ignored
                            if self.mode.startswith("immediate"):
                                self.fire(n)
                            else:
                                self.pending.append(n)
                                if self.mode.startswith("next+dup"):
                                    self.pending.append(("dup", n))
            elif t[4] == "air":
                self.redstone.discard(p)
            else:                                      # e.g. stone: a command block there is gone
                self.redstone.discard(p)
                self.blk.pop(p, None)
            return True
        if t[0] == "clone":                           # single-block clone keeps the command block data
            src, dst = tuple(int(v) for v in t[1:4]), tuple(int(v) for v in t[7:10])
            assert src == tuple(int(v) for v in t[4:7]), "only single-block clones are modelled"
            self.executed += 1; self.effective += 1
            if src in self.blk:
                b = dict(self.blk[src]); b["pos"] = list(dst); b.pop("last_ok", None)
                self.blk[dst] = b
            return True
        return super().run(cmd)

    def fire(self, head):
        self.fires[head] = self.fires.get(head, 0) + 1
        self.firing.add(head)
        try:
            self._fire(head)
        finally:
            self.firing.discard(head)

    def _fire(self, head):
        p = head
        self.forwards_run += 1
        steps = 0
        while True:
            steps += 1
            if steps == 3:                             # got past the 2nd block: a real forward
                self.full[head] = self.full.get(head, 0) + 1
            b = self.blk[p]
            d = DIRS[b["facing"]]
            if b["cond"]:
                behind = self.blk.get(tuple(p[j] - d[j] for j in range(3)))
                ok = bool(behind and behind.get("last_ok"))
            else:
                ok = True
            b["last_ok"] = bool(self.run(b["cmd"])) if ok else False
            n = tuple(p[j] + d[j] for j in range(3))
            nb = self.blk.get(n)
            if not nb or nb["kind"] != "chain":
                return
            p = n

    def tick(self):
        now, self.pending = self.pending, []
        is_dup = lambda x: isinstance(x[0], str)
        dups = [x[1] for x in now if is_dup(x)]
        work = [x for x in now if not is_dup(x)]
        for p, b in self.blk.items():
            if b["kind"] != "repeat":
                continue
            if b["auto"]:
                work.append(p)
            else:
                h = self.hist.setdefault(p, [])
                h.append(self.powered(p))
                if any(h[-(1 + self.lag):]):
                    work.append(p)
        self.rng.shuffle(work)
        for p in work:
            self.fire(p)
        self.pending += dups


def run_prompts(q, comp, plan, prompts, mode="next", lag=0, label=""):
    bad_seq = bad_state = 0
    ticks = forwards = stray = 0
    G = plan["power"].get("G")
    head = (G[0], G[1], G[2] + 1) if G else None       # single-chain head: forwards beyond fw are wasted
    for k, p in enumerate(prompts):
        im = S.IntModel(q)
        trace = []
        exp = im.generate(p, trace=trace)
        w = World(plan, mode, lag, seed=k)
        w.run_all(comp.init_program())
        w.run_all(comp.start_program(p) + plan["start_extra"])
        n = 0
        while w.s.get(("run", OBJ)) == 1 and n < 400:
            w.tick(); n += 1
        for _ in range(4):                                # stragglers (dups, lag) must not change results
            w.tick()
        got = [x for x in (w.s.get((f"out{i}", OBJ)) for i in range(S.T)) if x != -1]
        bad_seq += got != exp
        last = trace[-1]
        want = {snap(f"xf{S.L-1}x{i}"): int(last[f"xf{S.L-1}"][i]) for i in range(S.D)}
        want.update({snap("bt"): last["best"], snap("blg"): last["bestLog"]})
        bad_state += any(w.s.get((k2, OBJ)) != v for k2, v in want.items())
        ticks += n; forwards += w.s.get(("fw", OBJ))
        if head:
            stray += w.full.get(head, 0) - w.s.get(("fw", OBJ))   # forwards that ran the chain for nothing
        assert not w.faults, w.faults[:3]
    print(f"  {label:28s} prompts {len(prompts)}, sequence mismatches {bad_seq}, final-state mismatches {bad_state}, "
          f"ticks/forward {ticks / max(1, forwards):.2f}" + (f", wasted forwards {stray}" if head else ""))
    return bad_seq == 0 and bad_state == 0


def main(n=4, which="all", seed=0):
    q, comp, _ = build(seed)
    prompts = S.random_prompts(n, seed=seed + 11)
    ok = True
    if which in ("groups", "all"):
        plan = layout.build_plan(comp.groups())
        print("groups plan:", len(plan["blocks"]), "blocks", plan["chain_lengths"])
        for m in ("immediate", "next", "next+dup"):
            ok &= run_prompts(q, comp, plan, prompts, m, label=f"groups / impulse {m}")
    if which in ("single", "all"):
        plan = layout.build_plan_single(comp.groups(single=True), "repeat_rs")
        print("single (repeating head) plan:", len(plan["blocks"]), "blocks", plan["chain_lengths"])
        for lag in (0, 1, 2):
            ok &= run_prompts(q, comp, plan, prompts, lag=lag, label=f"single repeat / lag {lag}")
    if which in ("single_impulse", "all"):
        plan = layout.build_plan_single(comp.groups(single=True), "impulse")
        print("single (impulse head) plan:", len(plan["blocks"]), "blocks", plan["chain_lengths"])
        for m in ("next", "next+dup"):
            ok &= run_prompts(q, comp, plan, prompts, m, label=f"single impulse / {m}")
    if which in ("single_brk", "all"):
        plan = layout.build_plan_single(comp.groups(single=True), "repeat_brk")
        print("single (repeating head + chain breaker) plan:", len(plan["blocks"]), "blocks", plan["chain_lengths"])
        for lag in (0, 1, 2):
            ok &= run_prompts(q, comp, plan, prompts, lag=lag, label=f"single repeat+breaker / lag {lag}")
    if which in ("single_impulse_wd", "all"):
        plan = layout.build_plan_single(comp.groups(single=True), "impulse_wd")
        print("single (impulse head + watchdog) plan:", len(plan["blocks"]), "blocks", plan["chain_lengths"])
        for m in ("next-noreentry", "next", "immediate-noreentry"):
            ok &= run_prompts(q, comp, plan, prompts, m, label=f"single impulse+wd / {m}")
    print("WORLD SIM", "PASS" if ok else "FAIL")
    return ok


def main_staged(n=2, config="t44k", budget=8000):
    """staged layout (layout.build_plan_staged): stage chains relay each other through stg"""
    from t44k.check_staged import build as build_staged
    q, comp, stages = build_staged(config, budget=budget)
    plan = layout.build_plan_staged(stages)
    print(f"staged plan ({config}):", len(plan["blocks"]), "blocks,", len(stages), "stages")
    prompts = S.random_prompts(n, seed=11, max_len=3)
    ok = True
    for m in ("immediate", "next", "next+dup"):
        ok &= run_prompts(q, comp, plan, prompts, m, label=f"staged / impulse {m}")
    print("WORLD SIM (staged)", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) > 1 and a[1] == "staged":
        sys.exit(0 if main_staged(int(a[0]), a[2] if len(a) > 2 else "t44k",
                                  int(a[3]) if len(a) > 3 else 8000) else 1)
    sys.exit(0 if main(int(a[0]) if a else 4, a[1] if len(a) > 1 else "all") else 1)
