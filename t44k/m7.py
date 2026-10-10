"""M7: stand-alone T1M world with a 26-key PINYIN input (pure command blocks; no add-ons at runtime).

Near the player (buttons only): 26 letter keys (QWERTY rows), candidate keys 1-9, 上页 / 下页,
退格 / 清空 / 生成 / 自检 / 初始化 / 开机 / 关机. Every key only sets 'imk'; a far watchdog powers the IME chain.
Far away (the core's ticking area, above the single-chain T1M core of t44k/export_staged.py):
  IME     impulse chain, runs once per key press (no idle cost): pinyin-prefix state machine,
          candidate pick -> token 'in' -> appended to the prompt t1..t31, candidate display with
          titleraw actionbar, generation start; removes its own power at the end
  poller  repeating chain: the self-test check after a self-test run
  init    impulse chain: the export's init program (objectives, constants, KV cache) + IME state
State (objective t4; never remapped: the regmap only touches the core's bulk registers):
  imn pinyin trie node (0 = empty) · imp candidate page · imk key code of this press ·
  imx / imq next node / next page (all transitions read imn/imp and write imx/imq, so the order
  of commands inside the chain cannot cascade) · in picked token (-1 = none)
Key codes: letters a..z = 1..26, candidates 1..9 = 101..109, 下页 110, 上页 111, 退格 120,
清空 121, 生成 122, 自检 123.
Candidates of a node: characters whose (toneless) pinyin STARTS WITH the typed letters; exact
syllable matches first, then longer syllables, each by frequency in the training text (secondary
readings of polyphones weighted 0.3). Every character is reachable through its full syllable.

  python -m t44k.m7 EXPORT_DIR OUT_DIR [--dialog] [--model MODEL_DIR]   (EXPORT_DIR from export_staged ... single --trained --regmap)
  --dialog: dialogue model -- 发送 (key 124) appends <EOS> and starts / continues the conversation
"""
import sys, os, json, re, collections
import netease_lint
from pypinyin import pinyin, Style
from t44k import spec as S
from t44k import layout
from t44k.compiler import OBJ, sp, add, op, tell
from t44k.m6 import tellparts, keyboard_blocks, check_keyboard

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = 9
EXTRA_PAGES = 3          # prefix-only candidates (longer syllables) shown for a node: at most this many pages
QWERTY = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]


def ifs(*conds):
    """execute if score ... (conds: (name, range) pairs) run"""
    return "execute " + " ".join(f"if score {n} {OBJ} matches {r}" for n, r in conds) + " run "


def actionbar(text):
    return "titleraw @a actionbar " + json.dumps({"rawtext": [{"text": text}]}, ensure_ascii=False, separators=(",", ":"))


def char_freq(files, vocab):
    """character counts of the training text (only the vocabulary's characters): the candidate ranking.
    Released with the weights as char_freq.json, so the keyboard can be built without the corpus."""
    cs = set(vocab)
    c = collections.Counter(ch for f in files for l in open(f, encoding="utf-8") for ch in l.strip())
    return {ch: n for ch, n in c.items() if ch in cs}


def build_tables(vocab, train_txt):
    """train_txt: one text file or a list of them, or a {character: count} dict (character
    frequencies rank the candidates)"""
    idx = {c: i for i, c in enumerate(vocab)}
    chars = [c for c in vocab[4:] if "一" <= c <= "鿿"]
    if isinstance(train_txt, dict):
        freq = collections.Counter(train_txt)
    else:
        files = [train_txt] if isinstance(train_txt, str) else train_txt
        freq = collections.Counter(c for f in files for l in open(f, encoding="utf-8") for c in l.strip())
    reads = {c: pinyin(c, style=Style.NORMAL, heteronym=True)[0] for c in chars}
    weight = {}                                      # (char, syllable) -> ranking weight
    for c in chars:
        for k, s in enumerate(reads[c]):
            if re.fullmatch(r"[a-z]+", s):
                weight[(c, s)] = max(weight.get((c, s), 0), freq[c] * (1.0 if k == 0 else 0.3))
    syls = sorted({s for _, s in weight})
    prefixes = sorted({s[:i] for s in syls for i in range(1, len(s) + 1)}, key=lambda p: (len(p), p))
    node = {"": 0}
    for p in prefixes:
        node[p] = len(node)
    cands = {}
    for p, n in node.items():
        if not p:
            continue
        exact = sorted({c for (c, s) in weight if s == p}, key=lambda c: -weight[(c, p)])
        longer = {}
        for (c, s), w in weight.items():
            if s.startswith(p) and s != p and c not in exact:
                longer[c] = max(longer.get(c, 0), w)
        longer = sorted(longer, key=lambda c: -longer[c])
        n_extra = max(0, EXTRA_PAGES * PAGE - len(exact)) if exact else EXTRA_PAGES * PAGE
        cands[n] = [idx[c] for c in exact + longer[:n_extra]]
    return node, cands, set(syls)


def ime_chain(vocab, node, cands, syls, start_cmds, selftest, dialog=False, cont_cmds=None):
    """dialog=True: key 124 (发送) appends <EOS> and starts the reply; the first turn starts at
    pos 0 (start_cmds), later turns CONTINUE from the current position (cont_cmds: the KV cache of
    the conversation so far stays valid). State: cv conversation active, nt next character opens a
    new turn (it is written after the reply's <EOS> position: plen = pos + 1), ts turn start."""
    name = {n: p for p, n in node.items()}
    pages = {n: max(1, (len(c) + PAGE - 1) // PAGE) for n, c in cands.items()}
    pages[0] = 1
    c = [sp("in", -1), sp("imx", -1), op("imq", "=", "imp"), sp("ok", 0)]
    # letters: trie transitions
    for p, n in node.items():
        for L in "abcdefghijklmnopqrstuvwxyz":
            if p + L in node:
                c.append(ifs(("imk", ord(L) - 96), ("imn", n)) + sp("imx", node[p + L]))
    c.append(ifs(("imk", "1..26"), ("imx", "..-1")) + tell("§c[输入] 没有以这些字母开头的拼音"))
    # pages
    c += [ifs(("imk", 110)) + add("imq", 1), ifs(("imk", 111), ("imq", "1..")) + add("imq", -1)]
    # candidate picks (read imn / imp: unchanged so far)
    for n, lst in cands.items():
        for k, t in enumerate(lst):
            pg, slot = divmod(k, PAGE)
            c.append(ifs(("imk", 101 + slot), ("imn", n), ("imp", pg)) + sp("in", t))
    c.append(ifs(("imk", "101..109"), ("imn", 0)) + tell("§7[输入] 先按字母输入拼音，再按数字选字"))
    # backspace: inside a pinyin -> parent node; empty pinyin -> delete the last character
    for p, n in node.items():
        if p:
            c.append(ifs(("imk", 120), ("imn", n)) + sp("imx", node[p[:-1]]))
    if dialog:        # only within the current turn
        c += [f"execute if score imk {OBJ} matches 120 if score imn {OBJ} matches 0 if score nt {OBJ} matches 0 "
              f"if score plen {OBJ} > ts {OBJ} run " + add("plen", -1)]
    else:
        c += [ifs(("imk", 120), ("imn", 0), ("plen", "2..")) + add("plen", -1)]
    c += [ifs(("imk", 120), ("imn", 0)) + tell("§7[输入] 删除最后一个字")]
    # clear
    c += [ifs(("imk", 121)) + sp("plen", 1), ifs(("imk", 121)) + sp("imx", 0),
          ifs(("imk", 121)) + tell("§7[输入] 已清空" + ("，开始新的对话" if dialog else "提示词"))]
    if dialog:
        c += [ifs(("imk", 121)) + x for x in (sp("cv", 0), sp("nt", 0), sp("ts", 1))]
        # send: the turn must have characters and room for <EOS> + a reply
        c += [f"execute if score imk {OBJ} matches 124 if score run {OBJ} matches 0 if score nt {OBJ} matches 0 "
              f"if score plen {OBJ} > ts {OBJ} if score plen {OBJ} matches ..{S.T - 2} run " + sp("ok", 1),
              ifs(("imk", 124), ("ok", 0), ("run", 0)) + tell(f"§c[输入] 先输入要说的话（对话最多 {S.T} 字，满了请按清空）"),
              ifs(("imk", 124), ("run", 1)) + tell(f"§c[{S.label()}] 正在回答，请稍等"),
              ifs(("ok", 1), ("cv", 1)) + sp("ok", 2)]
        c += [ifs(("ok", "1..2"), ("plen", j)) + sp(f"t{j}", S.EOS) for j in range(1, S.T)]
        c += [ifs(("ok", "1..2")) + add("plen", 1)]
        c += [ifs(("ok", 1)) + x for x in start_cmds] + [ifs(("ok", 2)) + x for x in cont_cmds]
        c += [ifs(("ok", "1..2")) + x for x in (sp("cv", 1), sp("nt", 1), sp("imx", 0))]
        c += [ifs(("ok", "1..2")) + tell("§a[发送] 正在回答……")]
    # generate / self-test ('run' changes inside the start commands: latch the idle test into 'ok')
    c += [ifs(("imk", "122..123"), ("run", 0), ("plen", "2..")) + sp("ok", 1),
          ifs(("imk", 122), ("plen", "..1")) + tell("§c[输入] 还没有输入任何字"),
          ifs(("imk", "122..123"), ("run", 1)) + tell(f"§c[{S.label()}] 正在生成，请稍等")]
    st_prompt = selftest["prompt"]
    c += [ifs(("imk", 123), ("run", 0)) + x for x in
          [sp("plen", len(st_prompt))] + [sp(f"t{j}", t) for j, t in enumerate(st_prompt)] + [sp("ok", 1), sp("st", 1)]]
    c += [ifs(("imk", 123), ("ok", 1)) + tell(f"§d[{S.label()}] 自检：期望生成 {selftest['text']}")]
    if dialog:        # 'ok 1' here only comes from 自检 (122 does not exist on the dialogue keyboard)
        c += [ifs(("ok", 1), ("imk", 123)) + x for x in start_cmds]
        c += [ifs(("ok", 1), ("imk", 123)) + x for x in (sp("cv", 0), sp("nt", 0), sp("ts", len(st_prompt)), sp("imx", 0))]
    else:
        c += [ifs(("ok", 1)) + x for x in start_cmds] + [ifs(("ok", 1), ("imk", 122)) + tell(f"§a[{S.label()}] 开始生成……")]
        c += [ifs(("ok", 1)) + sp("imx", 0)]
    # apply the transition, clamp the page
    c += [ifs(("imx", "0..")) + op("imn", "=", "imx"), ifs(("imx", "0..")) + sp("imq", 0)]
    c += [ifs(("imn", n), ("imq", f"{pg}..")) + sp("imq", pg - 1) for n, pg in pages.items()]
    c.append(op("imp", "=", "imq"))
    # append the picked token to the prompt (t0 = BOS, plen counts it)
    if dialog:        # the first character of a new turn goes right after the reply's <EOS> position
        c += [ifs(("in", "0.."), ("run", 0), ("nt", 1)) + x for x in
              (op("plen", "=", "pos"), add("plen", 1), op("ts", "=", "plen"), sp("nt", 0))]
    c += [ifs(("in", "0.."), ("run", 1)) + tell(f"§c[{S.label()}] 正在生成，请稍等"),
          ifs(("in", "0.."), ("plen", f"{S.T}..")) + tell(f"§c[输入] 已满（最多 {S.T - 1} 个字）" + ("，请按清空开始新的对话" if dialog else ""))]
    c += [ifs(("in", "0.."), ("run", 0), ("plen", j)) + op(f"t{j}", "=", "in") for j in range(1, S.T)]
    c += [ifs(("in", "0.."), ("run", 0), ("plen", f"..{S.T - 1}")) + add("plen", 1)]
    for t in sorted({t for lst in cands.values() for t in lst}):
        c.append(ifs(("in", t), ("run", 0)) + tell(f"§7[输入] §e{vocab[t]}"))
    c += [ifs(("in", "0..")) + sp("imn", 0), ifs(("in", "0..")) + sp("imp", 0)]
    # display the candidates of the current node/page
    c.append(ifs(("imn", 0)) + actionbar("拼音输入：按字母键输入拼音，按 1-9 选字"))
    for n, lst in cands.items():
        for pg in range(pages[n]):
            items = "  ".join(f"{k + 1}{vocab[t]}" for k, t in enumerate(lst[pg * PAGE:(pg + 1) * PAGE]))
            mark = "" if name[n] in syls else "…"
            c.append(ifs(("imn", n), ("imp", pg)) + actionbar(f"{name[n]}{mark}  [{pg + 1}/{pages[n]}]   {items}"))
    return c


def build_m7(export_dir, dialog=False, model=None):
    """dialog=True: dialogue model (joint vocabulary data/mixd, candidates ranked on teacher +
    dialogue text) and the 发送 key instead of 生成.
    model: a model directory (vocab.json + char_freq.json, see check_staged.model_dir) instead of the
    data/ files"""
    meta = json.load(open(os.path.join(export_dir, "meta.json"), encoding="utf-8"))
    prog = json.load(open(os.path.join(export_dir, "programs.json"), encoding="utf-8"))
    assert meta["mode"].startswith("single_"), "M7 runs on a single-chain export (repeat_rs or impulse head)"
    S.configure(meta["config"])
    layout.set_footprint(192); layout.set_origin(meta["x0"])
    if model:
        vocab = json.load(open(os.path.join(model, "vocab.json"), encoding="utf-8"))["vocab"]
        node, cands, syls = build_tables(vocab, json.load(open(os.path.join(model, "char_freq.json"), encoding="utf-8")))
    elif dialog:
        vocab = json.load(open(os.path.join(ROOT, "data", "mixd", "vocab.json"), encoding="utf-8"))["vocab"]
        node, cands, syls = build_tables(vocab, [os.path.join(ROOT, "data", "mix64d", "train.txt"),
                                                 os.path.join(ROOT, "data", "dialog64e", "train.txt")])
    else:
        vocab = json.load(open(os.path.join(ROOT, "data", "mix", "vocab.json"), encoding="utf-8"))["vocab"]
        node, cands, syls = build_tables(vocab, os.path.join(ROOT, "data", "mix", "train.txt"))
    P = meta["power"]["G"]
    base_y = P[1] + layout.layers_for(meta["n_blocks"]) + 4          # above the core and its tick counter

    # start = the export's start program without the prompt part (plen / t*), + powering the core
    t0 = prog["tests"][0]
    start_cmds = [x for x in t0["start"] if not re.match(r"scoreboard players set (plen|t\d+) ", x)]
    # self-test: test prompt 0 of the export (its expected out* and final snapshot)
    first_gen = len(t0["prompt"]) - 1
    expect = {f"out{p}": -1 for p in range(S.T)}
    for k, t in enumerate(t0["out"]):
        expect[f"out{first_gen + k}"] = t
    expect.update(t0["final"])
    selftest = {"prompt": t0["prompt"], "text": t0["text"].replace("<EOS>", "<E>"), "checked": len(expect)}

    # continue (dialogue, later turns): the start without resetting the position / current token
    cont_cmds = [x for x in start_cmds if not re.match(r"scoreboard players (set pos |operation cur )", x)]
    blocks = []
    ime = ime_chain(vocab, node, cands, syls, start_cmds, selftest, dialog, cont_cmds)
    ime_power = (P[0], base_y, P[2])
    ime += [sp("imk", 0), f"setblock {ime_power[0]} {ime_power[1]} {ime_power[2]} air"]   # re-arm: remove own power
    blocks += layout.place_chain(ime, base_y, "impulse")
    # watchdog (repeating, always active; in the power row z = Z0, away from the power spot): keys only
    # set imk; while a key is pending, give the IME head a FRESH edge every tick (air, then redstone).
    # Keys used to place the redstone block themselves; in-game the power once stayed on without the
    # IME finishing (2026-10-02), so no later key produced an edge and the keyboard froze.
    IPs = "{} {} {}".format(*ime_power)
    wd = [f"execute if score imk {OBJ} matches 1.. run setblock {IPs} air",
          f"execute if score imk {OBJ} matches 1.. run setblock {IPs} redstone_block"]
    blocks += [{"pos": [P[0] + 4 + i, base_y, layout.Z0], "kind": "repeat" if i == 0 else "chain", "auto": True,
                "cond": False, "facing": 5, "cmd": c} for i, c in enumerate(wd)]
    base_y += layout.layers_for(len(ime)) + 1

    ck = f"execute if score st {OBJ} matches 1 if score run {OBJ} matches 0 "
    poll = [ck + "run " + sp("err", 0)]
    poll += [ck + f"unless score {n} {OBJ} matches {v} run " + add("err", 1) for n, v in expect.items()]
    poll += [ck + "run " + tellparts(f"§d[{S.label()}] 自检：比对 {len(expect)} 个整数，不一致 ", "$err", " 个"),
             ck + f"if score err {OBJ} matches 0 run " + tell(f"§a[{S.label()}] 自检 PASS：Minecraft 与 Python 精确模拟器逐整数一致"),
             ck + f"if score err {OBJ} matches 1.. run " + tell(f"§c[{S.label()}] 自检 FAIL"),
             ck + "run " + sp("st", 0)]
    blocks += layout.place_chain(poll, base_y, "repeat")
    base_y += layout.layers_for(len(poll)) + 1

    init = list(prog["init"])
    init += [sp(n, v) for n, v in (("in", -1), ("imn", 0), ("imp", 0), ("imk", 0), ("imx", -1), ("imq", 0), ("kp", 0),
                                   ("st", 0), ("err", 0), ("ok", 0), ("plen", 1), ("t0", S.BOS))]
    if dialog:
        init += [sp("cv", 0), sp("nt", 0), sp("ts", 1)]
    init += [tell(f"§a[{S.label()}] 初始化完成：{S.nominal_params():,} 参数的三值 Transformer，词表 {S.V}，上下文 {S.T}")]
    init_power = (P[0], base_y, P[2])
    init.append(f"setblock {init_power[0]} {init_power[1]} {init_power[2]} air")
    blocks += layout.place_chain(init, base_y, "impulse")
    base_y += layout.layers_for(len(init)) + 1
    assert base_y < 319, base_y

    # keyboard (relative coordinates; seen by the player facing south: x decreasing = to the right)
    press = lambda code: [sp("imk", code), add("kp", 1)]      # kp: key-press counter (diagnostics)
    keys = []
    for k in range(9):
        keys.append({"x": -k, "y": 8, "label": str(k + 1), "cmds": press(101 + k)})
    keys += [{"x": -10, "y": 8, "label": "上页", "cmds": press(111)}, {"x": -11, "y": 8, "label": "下页", "cmds": press(110)}]
    for r, row in enumerate(QWERTY):
        for k, L in enumerate(row):
            keys.append({"x": -k - r, "y": 6 - 2 * r, "label": L.upper(), "cmds": press(ord(L) - 96)})
    controls = [("退格", press(120)), ("清空", press(121)), ("发送", press(124)) if dialog else ("生成", press(122)),
                ("自检", press(123)),
                ("初始化", [f"setblock {init_power[0]} {init_power[1]} {init_power[2]} redstone_block",
                           tell(f"§7[{S.label()}] 初始化中……")]),
                # the far core must be in a ticking area to run; remove it (and the core's power) before
                # saving: a saved ticking area over ~900K command blocks stalls the loading screen.
                # Same name as ingame_staged.setup ("stga"): adding a second / renamed area over the
                # loaded core re-loads every chunk (2026-10-02: ~13 s per tick for 15+ minutes)
                ("开机", ["tickingarea add {} {} {} {} {} {} stga".format(*meta["tickingareas"][0]),
                          tell(f"§a[{S.label()}] 已加载常加载区（首次加载可能卡几分钟），然后按 初始化")]),
                # stop first (run 0): a pending forward then runs with live 0 and does not re-arm
                ("关机", [sp("run", 0), "setblock {} {} {} air".format(*P), "tickingarea remove stga",
                          tell(f"§a[{S.label()}] 已断电并移除常加载区，可以存档退出")])]
    for k, (label, cmds) in enumerate(controls):
        if label == "开机":       # far from 关机 (a mis-press re-loads ~900K command blocks: minutes of freeze)
            keys.append({"x": -14, "y": 8, "label": label, "cmds": cmds})
        else:
            keys.append({"x": -2 * len([1 for l, _ in controls[:k] if l != "开机"]), "y": 0, "label": label, "cmds": cmds})

    all_cmds = [b["cmd"] for b in blocks] + [x for k in keys for x in k["cmds"]]
    netease_lint.check(all_cmds, "m7")
    names = sorted({n for x in all_cmds for n in re.findall(r"(\S+) t4(?= |$)", x)
                    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", n)})
    netease_lint.check_names(names, "m7 names")
    stats = {"ime_commands": len(ime), "poll": len(poll), "init": len(init), "trie_nodes": len(node),
             "syllables": len(syls), "candidate_entries": sum(map(len, cands.values())),
             "pages": sum(max(1, (len(x) + PAGE - 1) // PAGE) for x in cands.values())}
    return {"chains": {"ime": ime, "poll": poll, "init": init, "watchdog": wd}, "tables": (node, cands, syls),
            "blocks": blocks, "keys": keys, "ime_power": list(ime_power), "init_power": list(init_power),
            "core_power": P, "tickingarea": meta["tickingareas"][0], "names": names, "selftest": selftest,
            "stats": stats, "model_hash": meta["model_hash"], "y_top": base_y}


if __name__ == "__main__":
    a = sys.argv[1:]
    mdir = None
    if "--model" in a:
        i = a.index("--model"); mdir = a[i + 1]; del a[i:i + 2]
    sys.argv[1:] = a
    m = build_m7(sys.argv[1], dialog="--dialog" in sys.argv, model=mdir)
    del m["chains"], m["tables"]                     # offline-test material (t44k/sim_m7.py)
    m["keyboard"] = keyboard_blocks(m["keys"])
    bad = check_keyboard(m["keyboard"])
    assert not bad, f"keyboard cross-talk: {bad[:5]}"
    os.makedirs(sys.argv[2], exist_ok=True)
    json.dump(m, open(os.path.join(sys.argv[2], "m7.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("far blocks", len(m["blocks"]), "keys", len(m["keys"]), "keyboard blocks", len(m["keyboard"]),
          "names", len(m["names"]), m["stats"], "self-test", m["selftest"]["text"], "y_top", m["y_top"])
