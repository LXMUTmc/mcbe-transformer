"""M6: stand-alone T44K world (pure command blocks, no functions, no add-ons at runtime).

Near the player (only buttons, cheap):
  key wall   one button per character (+ a sign with the character): sets 'in' and echoes it
  controls   清空 / 退格 / 生成 / 自检 / 初始化
Far away, in the core's ticking area (layout.X0 ...):
  core       the single-chain T44K forward (layout.build_plan_single)
  poller     always-active: appends 'in' to the prompt buffer t1..t15, runs the self-test check
  init       impulse chain: objectives, constants, KV cache, control registers (powered by 初始化)
Control buttons only set scores and place a redstone block next to a far impulse/repeating head.

Keyboard blocks use coordinates RELATIVE to the keyboard origin (chosen in-game next to the player);
everything else is absolute.

  python -m t44k.m6 OUT_DIR [x0]      (trained model: t44k/model/q.json + data/vocab_t44k.json)
"""
import sys, os, json, re
import netease_lint
from t44k import spec as S
from t44k import layout
from t44k.check_offline import build
from t44k.compiler import OBJ, sp, add, op, tell, snap

COLS = 64                 # keys per row
SELFTEST = "骷髅"          # fixed self-test prompt (two characters of the vocabulary)
SIGN_ALT = {"泽": "ze", "近": "j-i-n"}   # NetEase rejects these on signs (tellraw is fine); "jin" is rejected too


def tellparts(*parts):
    raw = []
    for p in parts:
        raw.append({"score": {"name": p[1:], "objective": OBJ}} if p.startswith("$") else {"text": p})
    return "tellraw @a " + json.dumps({"rawtext": raw}, ensure_ascii=False, separators=(",", ":"))


def build_m6(x0=16384):
    layout.set_origin(x0)
    q, comp, _ = build(model="trained")
    vocab = comp.vocab
    groups = comp.groups(single=True)
    core = layout.build_plan_single(groups, "repeat_rs")
    P = core["power"]["G"]
    PW = f"{P[0]} {P[1]} {P[2]}"
    base_y = core["y_range"][1] + 2
    blocks = list(core["blocks"])

    # ---------------------------------------------------------------- start (shared by 生成 / 自检)
    start = [sp("pos", 0), op("cur", "=", "t0")] + [sp(f"out{p}", -1) for p in range(S.T)] + \
            [sp("fw", 0), sp("run", 1), f"setblock {PW} redstone_block"]

    # ---------------------------------------------------------------- self-test expectations
    idx = {c: i for i, c in enumerate(vocab)}
    st_prompt = [S.BOS] + [idx[c] for c in SELFTEST]
    im = S.IntModel(q)
    trace = []
    st_out = im.generate(st_prompt, trace=trace)
    last = trace[-1]
    expect = {f"out{p}": -1 for p in range(S.T)}
    first_gen = len(st_prompt) - 1
    for k, t in enumerate(st_out):
        expect[f"out{first_gen + k}"] = t
    expect.update({snap(f"xf{S.L-1}x{i}"): int(last[f"xf{S.L-1}"][i]) for i in range(S.D)})
    expect.update({snap("bt"): int(last["best"]), snap("blg"): int(last["bestLog"])})
    st_text = SELFTEST + "→" + "".join("<EOS>" if t == S.EOS else vocab[t] for t in st_out)

    # ---------------------------------------------------------------- poller (always active)
    poll = []
    for j in range(1, S.T):
        poll.append(f"execute if score in {OBJ} matches 0.. if score run {OBJ} matches 0 if score plen {OBJ} matches {j} run "
                    + op(f"t{j}", "=", "in"))
    poll += [f"execute if score in {OBJ} matches 0.. if score run {OBJ} matches 0 if score plen {OBJ} matches ..{S.T - 1} run "
             + add("plen", 1),
             f"execute if score in {OBJ} matches 0.. if score plen {OBJ} matches {S.T}.. run " + tell("§c[T44K] 提示词已满（最多 15 个字）"),
             f"execute if score in {OBJ} matches 0.. if score run {OBJ} matches 1 run " + tell("§c[T44K] 正在生成，请稍等"),
             sp("in", -1)]
    # self-test check: after a self-test run ends (st == 1, run == 0)
    ck = f"execute if score st {OBJ} matches 1 if score run {OBJ} matches 0 "
    poll += [ck + "run " + sp("err", 0)]
    poll += [ck + f"unless score {n} {OBJ} matches {v} run " + add("err", 1) for n, v in expect.items()]
    poll += [ck + "run " + tellparts(f"§d[T44K] 自检：比对 {len(expect)} 个整数，不一致 ", "$err", " 个"),
             ck + f"if score err {OBJ} matches 0 run " + tell("§a[T44K] 自检 PASS：Minecraft 与 Python 精确模拟器逐整数一致"),
             ck + f"if score err {OBJ} matches 1.. run " + tell("§c[T44K] 自检 FAIL"),
             ck + "run " + sp("st", 0)]
    blocks += layout.place_chain(poll, base_y, "repeat")
    poll_y = base_y
    base_y += layout.layers_for(len(poll)) + 1

    # ---------------------------------------------------------------- init chain (impulse, far)
    init = [c for c in comp.init_program()]
    init += [sp("in", -1), sp("st", 0), sp("err", 0), sp("ok", 0), sp("plen", 1), sp("t0", S.BOS),
             tell("§a[T44K] 初始化完成：55,808 参数、词表 512、上下文 16 的三值 Transformer")]
    init = [c for c in init if not c.startswith("scoreboard objectives add")]   # objectives: first command below
    init = ["scoreboard objectives add t4 dummy", "scoreboard objectives add t4c dummy"] + init
    init.append(f"setblock {P[0]} {base_y} {P[2]} air")          # re-arm: remove own power
    blocks += layout.place_chain(init, base_y, "impulse")
    init_power = (P[0], base_y, P[2])
    base_y += layout.layers_for(len(init)) + 1
    assert base_y < 300

    # ---------------------------------------------------------------- keyboard (relative coordinates)
    keys = []
    chars = [i for i in range(4, S.V) if not vocab[i].startswith("<")]
    controls = [
        ("清空", [sp("plen", 1), sp("t0", S.BOS), tell("§7[T44K] 提示词已清空")]),
        ("退格", [f"execute if score plen {OBJ} matches 2.. run " + add("plen", -1), tell("§7[T44K] 删除最后一个字")]),
        # 'run' changes inside the chain (start sets run 1), so latch the idle test into 'ok' first
        ("生成", [sp("ok", 0), f"execute if score run {OBJ} matches 0 run " + sp("ok", 1)] +
                 [f"execute if score ok {OBJ} matches 1 run " + c for c in [sp("st", 0)] + start] +
                 [f"execute if score ok {OBJ} matches 1 run " + tell("§a[T44K] 开始生成……"),
                  f"execute if score ok {OBJ} matches 0 run " + tell("§c[T44K] 正在生成，请稍等")]),
        ("自检", [sp("ok", 0), f"execute if score run {OBJ} matches 0 run " + sp("ok", 1)] +
                 [f"execute if score ok {OBJ} matches 1 run " + c for c in
                  [sp("plen", len(st_prompt))] + [sp(f"t{j}", t) for j, t in enumerate(st_prompt)] + [sp("st", 1)] + start] +
                 [f"execute if score ok {OBJ} matches 1 run " + tell(f"§d[T44K] 自检：提示词「{SELFTEST}」，期望生成 {st_text}")]),
        ("初始化", [f"setblock {init_power[0]} {init_power[1]} {init_power[2]} redstone_block",
                   tell("§7[T44K] 初始化中……")]),
    ]
    # controls: bottom row; characters above, COLS per row; x DEcreases with index so that the
    # player (facing +z, i.e. south) reads left -> right
    rows = (len(chars) + COLS - 1) // COLS
    for k, (label, cmds) in enumerate(controls):
        keys.append({"x": -4 * k, "y": 0, "label": label, "cmds": cmds})
    for n, t in enumerate(chars):
        r, c = divmod(n, COLS)
        keys.append({"x": -c, "y": 3 + 2 * (rows - 1 - r), "label": vocab[t], "sign": SIGN_ALT.get(vocab[t], vocab[t]),
                     "cmds": [sp("in", t), tell("§7[输入] " + vocab[t])]})
    all_cmds = [b["cmd"] for b in blocks] + [c for k in keys for c in k["cmds"]]
    netease_lint.check(all_cmds, "m6")
    names = set()
    for c in all_cmds:
        names.update(re.findall(r"(\S+) (?:t4|t4c)(?= |$)", c))
    names = sorted(n for n in names if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", n))
    netease_lint.check_names(names, "m6 names")
    return {"blocks": blocks, "keys": keys, "tickingarea": core["tickingarea"], "core_power": P,
            "init_power": list(init_power), "poll_y": poll_y, "selftest": {"prompt": SELFTEST, "expected": st_text,
            "checked": len(expect)}, "names": names, "model_hash": S.qhash(q), "y_top": base_y,
            "chain_lengths": core["chain_lengths"]}


# ---------------------------------------------------------------- keyboard blocks (relative coordinates)
# Seen from the player (standing north of the wall, facing south = +z), per key at (x, y):
#   z = -1 : stone_button (facing north)        and the wall_sign one block higher (y + 1)
#   z =  0 : stone (the button's block)         and stone behind the sign (y + 1)
#   z =  1 : impulse command block (needs redstone), facing south
#   z >= 2 : always-active chain blocks for the key's remaining commands, facing south
# A button strongly powers the block it is attached to, and a strongly powered SOLID block
# activates every mechanism next to it. The first wall attached the buttons directly to the command
# blocks, so each press also fired the neighbouring keys' command blocks (cross-talk, user report
# 2026-09-30). With a stone in between, the stone only activates the one command block behind it:
# stone does not pass power to other stone, and an activated command block does not pass it on.
BUTTON, SIGN, CMD, STONE = "minecraft:stone_button", "minecraft:wall_sign", "minecraft:command_block", "minecraft:stone"
CHAIN = "minecraft:chain_command_block"
NORTH, SOUTH = 2, 3


def keyboard_blocks(keys):
    out = []
    for i, k in enumerate(keys):
        x, y = k["x"], k["y"]
        out += [{"rel": [x, y, -1], "name": BUTTON, "aux": NORTH, "key": i},
                {"rel": [x, y, 0], "name": STONE, "aux": 0},
                {"rel": [x, y + 1, 0], "name": STONE, "aux": 0},
                {"rel": [x, y + 1, -1], "name": SIGN, "aux": NORTH, "text": k.get("sign", k["label"])}]
        out += [{"rel": [x, y, 1 + j], "name": CMD if j == 0 else CHAIN, "aux": SOUTH, "cmd": c,
                 "mode": 0 if j == 0 else 2, "auto": j > 0, "key": i} for j, c in enumerate(k["cmds"])]
    return out


def check_keyboard(blocks):
    """Every button must activate exactly its own key's command block. Conservative redstone model:
    a pressed button activates the mechanisms next to itself and the block it is attached to; that
    block, if solid (stone or a command block), activates every command block next to it."""
    at = {tuple(b["rel"]): b for b in blocks}
    assert len(at) == len(blocks), "two blocks share a position"
    nb = lambda p: [(p[0] + dx, p[1] + dy, p[2] + dz) for dx, dy, dz in
                    ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
    back = {NORTH: (0, 0, 1), SOUTH: (0, 0, -1)}
    bad = []
    for b in blocks:
        if b["name"] != BUTTON:
            continue
        p = tuple(b["rel"])
        d = back[b["aux"]]
        a = (p[0] + d[0], p[1] + d[1], p[2] + d[2])
        assert a in at, f"button at {p} is not attached to anything"
        fired = {q for q in nb(p) if q in at and at[q]["name"] == CMD}
        if at[a]["name"] in (STONE, CMD):
            fired |= {q for q in nb(a) + [a] if q in at and at[q]["name"] == CMD}
        keys = sorted(at[q]["key"] for q in fired)
        if keys != [b["key"]]:
            bad.append((b["key"], keys))
    return bad


if __name__ == "__main__":
    out = sys.argv[1]
    m = build_m6(int(sys.argv[2]) if len(sys.argv) > 2 else 16384)
    m["keyboard"] = keyboard_blocks(m["keys"])
    bad = check_keyboard(m["keyboard"])
    assert not bad, f"keyboard cross-talk: {bad[:5]}"
    os.makedirs(out, exist_ok=True)
    json.dump(m, open(os.path.join(out, "m6.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("far blocks", len(m["blocks"]), "keys", len(m["keys"]), "keyboard blocks", len(m["keyboard"]), "names", len(m["names"]), "self-test", m["selftest"],
          "init power", m["init_power"], "y_top", m["y_top"], "model", m["model_hash"])
