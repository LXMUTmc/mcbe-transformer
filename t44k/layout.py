"""T44K world layout: tick-group chains + controller -> a placement plan.

The plan is plain data (JSON): every command block with position, kind, facing, conditional
flag and command. The same plan feeds the offline tick simulator (t44k/sim.py) and the in-game
builder (placed through mod-debug-bridge, t44k/ingame_place.py).

Placement (all inside one ticking area, far from the player: see CLAUDE.md render cost):
  x X0..X0+63, z Z0..Z0+63; z = Z0 row holds the power spots, chains snake over z 1..63.
  Group g: impulse head (needs redstone) at (X0, y_g, Z0+1), power spot (X0, y_g, Z0).
  Controller: always-active repeating chain in its own layer.
Group tail:   setblock <own power> air ; [grp==g] arm = 1 ; [grp==g] grp = next
Controller:   tk += 1 ; [run==1, arm==1, grp==g] setblock <P_g> redstone_block ; [run==1, arm==1] arm = 0
"""
from t44k.compiler import OBJ, guard, flatten, sp, add

X0, Z0, Y0 = 8192, 0, 100
W, D = 64, 63
FACE = {(0, -1, 0): 0, (0, 1, 0): 1, (0, 0, -1): 2, (0, 0, 1): 3, (-1, 0, 0): 4, (1, 0, 0): 5}
TICKAREA = (X0, -64, Z0, X0 + 63, 319, Z0 + 63)


def set_origin(x0):
    """move the whole layout (e.g. to a clean region of an existing test world)"""
    global X0, TICKAREA
    X0 = x0
    TICKAREA = (X0, -64, Z0, X0 + W - 1, 319, Z0 + 63)


def set_footprint(w):
    """snake width in x (default 64). A wider footprint keeps a very long single chain low
    (T1M: 192 x 63 per layer, 3 x 4 chunks = one ticking area of 12 chunks... per 64 columns)"""
    global W
    W = w
    set_origin(X0)


def snake(y0, n):
    cells = []
    for k in range(n):
        layer = []
        for zi, z in enumerate(range(1, D + 1)):
            xs = range(W) if zi % 2 == 0 else range(W - 1, -1, -1)
            layer += [(x, y0 + k, z) for x in xs]
        if k % 2 == 1:
            layer.reverse()
        cells += layer
    return cells


def layers_for(n):
    return n // (W * D) + 1


def regate(cmds):
    """Bedrock conditional blocks test the block GEOMETRICALLY BEHIND them (opposite their
    facing), not their chain predecessor. At every snake turn two cells break that assumption
    (measured in-game: wrong embedding dims). Put the current segment's gate (the last
    unconditional command) there instead, as an unconditional block; gates only read cur/pos,
    which do not change inside a group, so the result is identical."""
    out, gate = [], None
    ci = 0                                       # index of the cell the next command lands on
    cells = snake(0, layers_for(len(cmds) * 2 + 8))

    def broken(k):
        if k == 0:
            return False
        p, prev, nx = cells[k], cells[k - 1], cells[k + 1]
        facing = tuple(nx[j] - p[j] for j in range(3))
        return tuple(p[j] - facing[j] for j in range(3)) != prev

    for c in cmds:
        if c.startswith("?"):
            while broken(ci):
                assert gate is not None, "conditional block without a preceding gate"
                out.append(gate); ci += 1
        else:
            gate = c
        out.append(c); ci += 1
    return out


def place_chain(cmds, y0, head, x0=None):
    """head: 'impulse' (needs redstone) | 'repeat' (always active). Returns list of block dicts."""
    x0 = X0 if x0 is None else x0
    cmds = regate(cmds)
    cells = snake(y0, layers_for(len(cmds)))
    out = []
    for i, c in enumerate(cmds):
        p, nx = cells[i], cells[i + 1]
        conditional = c.startswith("?")
        if conditional:
            assert i > 0, "a head cannot be conditional"
            c = c[1:]
        if i == 0:
            kind = "impulse" if head == "impulse" else "repeat"
            auto = head == "repeat"                   # 'repeat_rs' = repeating, needs redstone
        else:
            kind, auto = "chain", True
        out.append({"pos": [x0 + p[0], p[1], Z0 + p[2]], "kind": kind, "auto": auto, "cond": conditional,
                    "facing": FACE[tuple(nx[j] - p[j] for j in range(3))], "cmd": c})
    return out


def build_plan(groups):
    """groups: [(name, sections)] from Compiler.groups(). Returns plan dict."""
    y = Y0
    ys, chains = {}, {}
    for gi, (name, secs) in enumerate(groups):
        n = len(regate(flatten(secs) + ["tail1", "tail2", "tail3"]))
        ys[name] = y
        y += layers_for(n) + 1
    ctrl_y = y
    power = {name: [X0, ys[name], Z0] for name in ys}
    names = [n for n, _ in groups]
    blocks = []
    for gi, (name, secs) in enumerate(groups):
        nxt = (gi + 1) % len(groups)
        p = power[name]
        tail = [f"setblock {p[0]} {p[1]} {p[2]} air",
                guard("grp", gi, sp("arm", 1)),
                guard("grp", gi, sp("grp", nxt))]
        cmds = flatten(secs) + tail
        chains[name] = len(regate(cmds))
        blocks += place_chain(cmds, ys[name], "impulse")
    # take the 'armed' token first (go), THEN power: if an impulse head fires immediately inside
    # the setblock, the whole token may run in this tick and re-arm; nothing here may clobber that.
    ctrl = [add("tk", 1), sp("go", 0),
            f"execute if score run {OBJ} matches 1 if score arm {OBJ} matches 1 run " + sp("go", 1),
            f"execute if score go {OBJ} matches 1 run " + sp("arm", 0)]
    for gi, name in enumerate(names):
        p = power[name]
        ctrl.append(f"execute if score go {OBJ} matches 1 if score grp {OBJ} matches {gi} "
                    f"run setblock {p[0]} {p[1]} {p[2]} redstone_block")
    blocks += place_chain(ctrl, ctrl_y, "repeat")
    top = ctrl_y + 1
    assert top < 319
    return {"mode": "groups", "tickingarea": list(TICKAREA), "power": power, "group_y": ys, "controller_y": ctrl_y,
            "chain_lengths": chains, "controller_len": len(ctrl), "blocks": blocks,
            "y_range": [Y0, top], "start_extra": [], "stop_extra": []}


def build_plan_staged(stages, y_lo=-60, y_hi=36, max_chain=65535):
    # y_hi: 96 layers per column ~ 24K command blocks per chunk (E4 reopened a world at ~19K/chunk)
    # columns are W wide (set_footprint); max_chain: raise with gamerule maxcommandchainlength
    """one impulse chain per stage (Compiler.stages()); the stages relay each other:
      tail of stage k: setblock <own power> air ; go = [stg==k] ; [go] stg = k+1 ; [go] setblock <power k+1> redstone_block
      last stage     : the same with stg = 0 and power 0, only while run == 1
    A chain that fires a second time finds stg != k and relays nothing. Chains fill 64-wide
    columns bottom-up; each column is its own ticking area (Bedrock allows 10)."""
    col, y = 0, y_lo
    where, chains = [], {}
    n_st = len(stages)
    for k, (name, secs) in enumerate(stages):
        n = len(regate(flatten(secs) + ["t1", "t2", "t3", "t4", "t5"]))
        assert n <= max_chain, f"{name}: chain too long ({n})"
        if y + layers_for(n) > y_hi:
            col, y = col + 1, y_lo
        assert y + layers_for(n) <= y_hi, f"{name}: does not fit in one column"
        where.append((X0 + W * col, y))
        y += layers_for(n) + 1
    power = {name: [where[k][0], where[k][1], Z0] for k, (name, _) in enumerate(stages)}
    P = lambda k: "{} {} {}".format(*power[stages[k][0]])
    blocks = []
    for k, (name, secs) in enumerate(stages):
        # latch 'stg == k' into go, advance stg, and power the next stage LAST: if an impulse fires
        # inside the setblock, the next stage must already see its own stg value
        nxt = k + 1 if k < n_st - 1 else 0
        tail = [f"setblock {P(k)} air", sp("go", 0),
                f"execute if score stg {OBJ} matches {k} run " + sp("go", 1),
                f"execute if score go {OBJ} matches 1 run " + sp("stg", nxt),
                f"execute if score go {OBJ} matches 1 " + (f"if score run {OBJ} matches 1 " if nxt == 0 else "")
                + f"run setblock {P(nxt)} redstone_block"]
        cmds = flatten(secs) + tail
        chains[name] = len(regate(cmds))
        blocks += place_chain(cmds, where[k][1], "impulse", where[k][0])
    if y + 2 > y_hi:
        col, y = col + 1, y_lo
    blocks += place_chain([add("tk", 1)], y, "repeat", X0 + W * col)       # tick counter
    areas = [[X0 + W * c, -64, Z0, X0 + W * c + W - 1, 319, Z0 + 63] for c in range(col + 1)]
    assert len(areas) <= 10, "Bedrock allows 10 ticking areas"
    return {"mode": "staged", "tickingareas": areas, "tickingarea": areas[0], "power": power,
            "chain_lengths": chains, "blocks": blocks, "columns": col + 1,
            "start_extra": [f"setblock {P(0)} air", f"setblock {P(0)} redstone_block"],
            "stop_extra": [f"setblock {P(k)} air" for k in range(n_st)]}


def build_plan_single(groups, head="repeat_rs", max_chain=65535):
    """ONE chain = one full forward (<= 65535 blocks). groups must come from
    Compiler.groups(single=True) (live latch + live-guarded control).
      head 'repeat_rs': repeating head that needs redstone -> one forward per tick while powered;
                        the chain removes its own power when run drops to 0.
      head 'impulse'  : the chain re-arms itself (air, then redstone if run == 1) -> a fresh edge."""
    p = [X0, Y0, Z0]
    cmds = [c for _, secs in groups for c in flatten(secs)]
    P = f"{p[0]} {p[1]} {p[2]}"
    wd, extra = [], []
    if head == "repeat_rs":
        tail = [f"execute if score run {OBJ} matches 0 run setblock {P} air"]
        start = [f"setblock {P} redstone_block"]
    elif head == "repeat_brk":
        # repeating head (one forward per tick, proven) + chain BREAKER: a repeating head keeps running
        # 1-2 ticks after its power is removed (7-14 s of wasted forwards on T4M). When run drops to 0
        # the tail replaces the 2nd block of the chain with stone, so those ticks only run the head;
        # the start clones the 2nd block back from a template copy (clone keeps the command block
        # data). Measured in-game 2026-10-02 (T4M): 1 tick per forward, nothing runs after the end.
        B2 = [X0 + 1, Y0, Z0 + 1]
        TPL = [X0 + 12, Y0, Z0]
        tail = [f"execute if score run {OBJ} matches 0 run setblock {P} air",
                f"execute if score run {OBJ} matches 0 run setblock {B2[0]} {B2[1]} {B2[2]} stone"]
        start = [f"clone {TPL[0]} {TPL[1]} {TPL[2]} {TPL[0]} {TPL[1]} {TPL[2]} {B2[0]} {B2[1]} {B2[2]}",
                 f"setblock {P} redstone_block"]
    elif head == "impulse":
        # NOTE: in NetEase a chain re-powering its OWN impulse head is ignored (measured 2026-10-02:
        # the forward stopped after one token). Use 'impulse_wd'.
        tail = [f"setblock {P} air", f"execute if score run {OBJ} matches 1 run setblock {P} redstone_block"]
        start = [f"setblock {P} air", f"setblock {P} redstone_block"]
    else:   # 'impulse_wd' -- FAILS in NetEase (measured 2026-10-02: the head never fired again, it seems
        # to keep a stale 'powered' state when its own chain removed the power); kept for the simulator.
        # The tail only requests the next forward (go = 1); a separate always-active
        # watchdog gives the impulse head a fresh edge (air, then redstone) and clears the request.
        # No forward runs after the last one (a repeating head runs 1-2 more: 7-14 s on T4M).
        tail = [f"setblock {P} air", f"execute if score run {OBJ} matches 1 run " + sp("go", 1)]
        start = [sp("go", 1)]
        # take the request first (gw), so a head that fires INSIDE the setblock can post the next one
        wd = [sp("gw", 0), f"execute if score go {OBJ} matches 1 run " + sp("gw", 1), sp("go", 0),
              f"execute if score gw {OBJ} matches 1 run setblock {P} air",
              f"execute if score gw {OBJ} matches 1 run setblock {P} redstone_block"]
    chain = cmds + tail
    blocks = place_chain(chain, Y0, "impulse" if head.startswith("impulse") else
                         "repeat_rs" if head == "repeat_brk" else head)
    if head == "repeat_brk":                       # template copy of the 2nd block (never triggered)
        b2 = dict(next(b for b in blocks if b["pos"] == B2))
        b2["pos"] = TPL
        blocks.append(b2)
    # watchdog in the power row (z = Z0), not next to the power spot
    blocks += [{"pos": [X0 + 4 + i, Y0, Z0], "kind": "repeat" if i == 0 else "chain", "auto": True, "cond": False,
                "facing": FACE[(1, 0, 0)], "cmd": c} for i, c in enumerate(wd)]
    n = len(regate(chain))
    assert n <= max_chain, f"chain too long for maxcommandchainlength: {n}"   # measured: the gamerule can be raised to 1e6
    ctr_y = Y0 + layers_for(n) + 1
    blocks += place_chain([add("tk", 1)], ctr_y, "repeat")
    return {"mode": "single_" + head, "tickingarea": list(TICKAREA), "power": {"G": p}, "chain_lengths": {"G": n},
            "blocks": blocks, "y_range": [Y0, ctr_y + 1], "start_extra": start,
            "stop_extra": [f"setblock {P} air"]}
