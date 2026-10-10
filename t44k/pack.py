"""Behavior pack installer for a staged build: no developer tools, works on vanilla Bedrock and NetEase.

Input : an export_staged directory (blocks.txt, meta.json, programs.json) and, optionally, the M7
        keyboard build (m7.json from t44k.m7).
Output: OUT_DIR/behavior_pack/        structures/mct/<tag>_NN.mcstructure  the far core + M7 parts, in
                                      64x64x64 tiles (absolute positions; unset cells = structure void)
                                      structures/mct/<tag>_kb.mcstructure  the keyboard (relative)
                                      functions/mct/*.mcfunction, functions/tick.json
        OUT_DIR/<tag>.mcpack          the pack for vanilla Bedrock (double-click to import)
        OUT_DIR/<tag>_netease.zip     the same pack as a NetEase MC Studio component
                                      (root = behavior_pack_* + resource_pack_*)

In the game (a NEW flat world, cheats on, the pack enabled):
  /function mct/install    gamerules, ticking area "stga" over the far core, then the pack's tick
                           function loads one tile per tick once its chunks are loaded (the game
                           freezes for a few seconds per tile)
  /function mct/keyboard   the keyboard in front of the player (stand where it should go, face SOUTH)
Then press 初始化 on the keyboard. The pack can be removed afterwards: the world is plain vanilla
blocks (the tick function only runs while an installation is in progress).

Command block NBT follows what the game itself writes (read back in-game, NetEase 1.21.120:
Version 44, LPCondionalMode = conditional) plus conditionalMode, which the game uses on load to set
the block's conditional bit (the palette state alone is overridden); TrackOutput 0.

  python -m t44k.pack EXPORT_DIR [M7_DIR] OUT_DIR [--tag d10f] [--name "MCBE Transformer T10M"]
  python -m t44k.pack --verify OUT_DIR EXPORT_DIR [M7_DIR]     read every tile back and compare
"""
import sys, os, json, struct, zipfile, hashlib, uuid, shutil, math, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import netease_lint
from nbt import loads

TILE = 64
CB_VERSION = 44                    # what NetEase 1.21.120 writes for command blocks
BLOCK_VERSION = 18087937           # block-state version 1.20.0.1 (upgraded by newer games on load)
NAME = {"impulse": "minecraft:command_block", "repeat": "minecraft:repeating_command_block",
        "chain": "minecraft:chain_command_block"}
MODE = {"impulse": 0, "repeat": 1, "chain": 2}
PACK_VERSION = [1, 0, 0]


# ---------------------------------------------------------------- little-endian NBT, fast path
def _s(b):
    return struct.pack("<H", len(b)) + b


def _key(t, name):
    return bytes([t]) + _s(name.encode())


def t_byte(n, v): return _key(1, n) + struct.pack("<b", v)
def t_int(n, v): return _key(3, n) + struct.pack("<i", v)
def t_long(n, v): return _key(4, n) + struct.pack("<q", v)
def t_str(n, v): return _key(8, n) + _s(v.encode("utf-8"))
def t_comp(n, body): return _key(10, n) + body + b"\x00"


_CB_FIXED = (t_str("id", "CommandBlock") + t_str("CustomName", "") + t_byte("ExecuteOnFirstTick", 0)
             + t_long("LastExecution", 0) + t_str("LastOutput", "") + _key(9, "LastOutputParams") + struct.pack("<bi", 8, 0)
             + t_int("SuccessCount", 0) + t_int("TickDelay", 0) + t_byte("TrackOutput", 0)
             + t_int("Version", CB_VERSION) + t_byte("conditionMet", 0) + t_byte("isMovable", 1) + t_byte("powered", 0))


def cb_nbt(cmd, kind, cond, auto, x, y, z):
    # conditionalMode: on load the game re-sets the block's conditional_bit from THIS field (absent = 0;
    # measured 2026-10-09: 1.3M conditional blocks came out unconditional without it)
    return (_CB_FIXED + t_str("Command", cmd) + t_int("LPCommandMode", MODE[kind]) + t_byte("LPCondionalMode", int(cond))
            + t_byte("conditionalMode", int(cond))
            + t_byte("LPRedstoneMode", 0 if auto else 1) + t_byte("auto", int(auto))
            + t_int("x", x) + t_int("y", y) + t_int("z", z))


def sign_nbt(text, x, y, z):
    face = lambda t: (t_str("Text", t) + t_str("TextOwner", "") + t_int("SignTextColor", -16777216) + t_byte("IgnoreLighting", 0)
                      + t_byte("HideGlowOutline", 0) + t_byte("PersistFormatting", 1) + t_str("FilteredText", ""))
    return (t_str("id", "Sign") + t_comp("FrontText", face(text)) + t_comp("BackText", face("")) + t_byte("IsWaxed", 0)
            + t_byte("isMovable", 1) + t_int("x", x) + t_int("y", y) + t_int("z", z))


def states_nbt(states):
    out = b""
    for k, v in sorted(states.items()):
        out += t_byte(k, v[1]) if v[0] == "b" else t_int(k, v[1])
    return out


def structure_bytes(size, cells):
    """size (sx, sy, sz); cells: {(x, y, z): (block name, states {k: ('b'|'i', v)}, entity NBT body or None)}
    with positions inside the structure. Unset cells are structure void."""
    sx, sy, sz = size
    n = sx * sy * sz
    idx = [-1] * n
    palette, pidx, posdata = [], {}, bytearray()
    for (x, y, z), (name, states, ent) in cells.items():
        key = (name, tuple(sorted(states.items())))
        if key not in pidx:
            pidx[key] = len(palette)
            palette.append(t_str("name", name) + t_comp("states", states_nbt(states)) + t_int("version", BLOCK_VERSION))
        i = (x * sy + y) * sz + z                   # z fastest, then y, then x
        idx[i] = pidx[key]
        if ent is not None:
            posdata += t_comp(str(i), t_comp("block_entity_data", ent))
    out = bytearray(b"\x0a" + _s(b""))
    out += t_int("format_version", 1)
    out += _key(9, "size") + struct.pack("<bi", 3, 3) + struct.pack("<3i", sx, sy, sz)
    out += _key(10, "structure")
    out += _key(9, "block_indices") + struct.pack("<bi", 9, 2)
    out += struct.pack("<bi", 3, n) + struct.pack("<%di" % n, *idx)
    out += struct.pack("<bi", 3, n) + struct.pack("<%di" % n, *([-1] * n))
    out += _key(9, "entities") + struct.pack("<bi", 10, 0)
    out += _key(10, "palette") + _key(10, "default")
    out += _key(9, "block_palette") + struct.pack("<bi", 10, len(palette)) + b"".join(p + b"\x00" for p in palette)
    out += _key(10, "block_position_data") + posdata + b"\x00"
    out += b"\x00\x00"                              # default, palette
    out += b"\x00"                                  # structure
    out += _key(9, "structure_world_origin") + struct.pack("<bi", 3, 3) + struct.pack("<3i", 0, 0, 0)
    out += b"\x00"
    return bytes(out)


# ---------------------------------------------------------------- inputs
def read_blocks(export_dir, m7_dir=None):
    """every far block as (x, y, z) -> (kind, facing, cond, auto, cmd)"""
    blocks = {}
    with open(os.path.join(export_dir, "blocks.txt"), encoding="utf-8") as f:
        for line in f:
            head, cmd = line.rstrip("\n").split("\t", 1)
            x, y, z, kind, facing, cond, auto = head.split(" ")
            blocks[(int(x), int(y), int(z))] = (kind, int(facing), cond in ("1", "True"), auto in ("1", "True"), cmd)
    m7 = None
    if m7_dir:
        m7 = json.load(open(os.path.join(m7_dir, "m7.json"), encoding="utf-8"))
        for b in m7["blocks"]:
            p = tuple(b["pos"])
            assert p not in blocks, f"M7 block on the core at {p}"
            blocks[p] = (b["kind"], b["facing"], bool(b["cond"]), bool(b["auto"]), b["cmd"])
    return blocks, m7


def cb_cell(kind, facing, cond, auto, cmd, rel):
    return (NAME[kind], {"facing_direction": ("i", facing), "conditional_bit": ("b", int(cond))},
            cb_nbt(cmd, kind, cond, auto, *rel))


def tiles_of(blocks):
    """64^3 tiles (origin, size, {rel pos: block}) covering the blocks, aligned to the lowest corner"""
    lo = [min(p[i] for p in blocks) for i in range(3)]
    hi = [max(p[i] for p in blocks) for i in range(3)]
    buckets = {}
    for p, b in blocks.items():
        k = tuple((p[i] - lo[i]) // TILE for i in range(3))
        buckets.setdefault(k, {})[p] = b
    out = []
    for k in sorted(buckets, key=lambda k: (k[1], k[0], k[2])):          # bottom layer first
        org = [lo[i] + k[i] * TILE for i in range(3)]
        size = [min(TILE, hi[i] - org[i] + 1) for i in range(3)]
        out.append((org, size, buckets[k]))
    return out


def keyboard_cells(m7):
    """the M7 keyboard as relative cells (shifted so the minimum is 0) and that minimum (the offset
    from t44k/m7_ingame.place_wall: key rel + (x + 5, y, z + 3) for a player at (x, y, z))"""
    kb = m7["keyboard"]
    lo = [min(b["rel"][i] for b in kb) for i in range(3)]
    cells = {}
    for b in kb:
        p = tuple(b["rel"][i] - lo[i] for i in range(3))
        name = b["name"]
        if "command_block" in name:
            kind = "chain" if "chain" in name else ("repeat" if "repeating" in name else "impulse")
            assert kind == {0: "impulse", 1: "repeat", 2: "chain"}[b["mode"]]
            cells[p] = cb_cell(kind, b["aux"] & 7, bool(b["aux"] & 8), bool(b["auto"]), b["cmd"], p)
        elif name == "minecraft:stone_button":
            cells[p] = (name, {"facing_direction": ("i", b["aux"]), "button_pressed_bit": ("b", 0)}, None)
        elif name == "minecraft:wall_sign":
            cells[p] = (name, {"facing_direction": ("i", b["aux"])}, sign_nbt(b["text"], *p))
        else:
            assert b.get("aux", 0) == 0 and "cmd" not in b and "text" not in b, b
            cells[p] = (name, {}, None)
    return cells, lo


# ---------------------------------------------------------------- functions
def tellraw(text):
    return 'tellraw @a {"rawtext":[{"text":"%s"}]}' % text


def functions(tag, tiles, meta, chain_rule, kb_offset, n_blocks):
    ta = meta["tickingareas"][0]
    k = len(tiles)
    install = ["gamerule commandblockoutput false", f"gamerule maxcommandchainlength {chain_rule}",
               "tickingarea add {} {} {} {} {} {} stga".format(*ta),
               "scoreboard objectives add mctl dummy", "scoreboard players set i mctl 0", "scoreboard players set on mctl 1",
               tellraw(f"§e[MCT] 开始安装：{n_blocks:,} 个方块，分 {k} 批加载；每批游戏会卡住几秒到几十秒，属于正常现象，请不要关闭游戏")]
    load = []
    for i, (org, size, _) in enumerate(tiles):
        corners = {(org[0], org[2]), (org[0] + size[0] - 1, org[2]), (org[0], org[2] + size[2] - 1),
                   (org[0] + size[0] - 1, org[2] + size[2] - 1)}
        # drop the game's in-memory copy of the PREVIOUS tile's template, at least a tick after it was
        # placed (reports failure for pack structures but frees the cache: measured 2026-10-09, T10M game
        # process 18.3 -> 11.2 GB after deleting all 15)
        if i:
            load.append(f"execute if score i mctl matches {i} run structure delete mct:{tag}_{i - 1:02d}")
        load.append(f"execute if score i mctl matches {i} run scoreboard players set ok mctl 0")
        # a chunk is loaded when its bedrock floor (y = -64) can be read; until then wait
        load += [f"execute if score i mctl matches {i} if block {x} -64 {z} bedrock run scoreboard players add ok mctl 1"
                 for x, z in sorted(corners)]
        load.append(f"execute if score i mctl matches {i} if score ok mctl matches {len(corners)} run "
                    f"structure load mct:{tag}_{i:02d} {org[0]} {org[1]} {org[2]}")
        load.append(f"execute if score i mctl matches {i} if score ok mctl matches {len(corners)} run "
                    + tellraw(f"§7[MCT] 已加载 {i + 1}/{k}"))
        load.append(f"execute if score i mctl matches {i} if score ok mctl matches {len(corners)} run scoreboard players set ok mctl -1")
    load.append("execute if score ok mctl matches -1 run scoreboard players add i mctl 1")
    # finish one tick after the last tile (done deletes its cached template)
    load.insert(0, f"execute if score i mctl matches {k}.. run function mct/done")
    done = ["scoreboard players set on mctl 0", f"structure delete mct:{tag}_{k - 1:02d}",
            tellraw("§a[MCT] 模型已装好。走到想放键盘的空地，面朝南，输入 /function mct/keyboard"),
            tellraw("§7[MCT] 键盘放好后按 初始化；之后可以移除这个行为包，世界只剩原版方块")]
    keyboard = [f"structure load mct:{tag}_kb ~{kb_offset[0]} ~{kb_offset[1]} ~{kb_offset[2]}",
                tellraw("§a[MCT] 键盘已放好：先按 初始化，然后用字母键打拼音、数字键选字，按 发送"),
                tellraw("§7[MCT] 每次按键间隔约 3 秒；存档退出前先按 关机")]
    tick = ["execute if score on mctl matches 1 run function mct/load"]
    fs = {"install": install, "load": load, "done": done, "keyboard": keyboard, "tick": tick}
    for name, cmds in fs.items():
        netease_lint.check(cmds, f"pack function {name}")
    return fs


def auto_functions(info):
    """auto-install variant (pack.py --auto, and every world from t44k/world_build.py): the pack installs
    itself the first time a world with it is opened, then places the keyboard where the nearest player
    stands (as /function mct/keyboard would) and presses 初始化. Overrides tick / done, adds autostart."""
    tag, k = info["tag"], len(info["tiles"])
    off, ip = info["keyboard_offset"], info["init_power"]
    assert ip, "auto install needs the keyboard (build the pack with an M7 directory)"
    # Start only after a player has been in the world for WAIT ticks: on NetEase the integrated server
    # ticks while the client is still loading the world, and starting the installer then (ticking area +
    # 260K command blocks per tick) crashed the game twice during the loading screen (2026-10-10).
    WAIT = 300
    tick = ["scoreboard objectives add mctl dummy",           # fails harmlessly once it exists
            "scoreboard players add boot mctl 0",
            f"execute if score boot mctl matches 0..{WAIT - 1} if entity @a run scoreboard players add boot mctl 1",
            "execute if score boot mctl matches 1 run " + tellraw(f"§e[MCT] 首次进入：{WAIT // 20} 秒后开始自动安装模型（期间游戏会卡住十几次，请不要关闭游戏）"),
            f"execute if score boot mctl matches {WAIT} run function mct/autostart",
            "execute if score on mctl matches 1 run function mct/load"]
    autostart = [f"scoreboard players set boot mctl {WAIT + 1}", "function mct/install"]
    done = ["scoreboard players set on mctl 0", f"structure delete mct:{tag}_{k - 1:02d}",
            "execute as @p at @s run structure load mct:{}_kb ~{} ~{} ~{}".format(tag, *off),
            "setblock {} {} {} redstone_block".format(*ip),
            tellraw("§a[MCT] 模型已装好，键盘放在你前方（南边）。聊天栏出现“初始化完成”后就可以聊天了"),
            tellraw("§7[MCT] 用法：字母键打拼音，数字键选字，按 发送；看到屏幕下方提示变了再按下一个键；退出前先按 关机"),
            tellraw("§7[MCT] 想把键盘放到别处：站好、面朝南，输入 /function mct/keyboard")]
    fs = {"tick": tick, "autostart": autostart, "done": done}
    for n, c in fs.items():
        netease_lint.check(c, f"auto function {n}")
    return fs


# ---------------------------------------------------------------- build
def _uuid(s):
    return str(uuid.UUID(hashlib.md5(s.encode()).hexdigest()))


def zip_dir(src, zpath, prefix=""):
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, _, files in os.walk(src):
            rel = os.path.relpath(root, src)
            if rel != ".":       # directory entries too: MC Studio wants the (empty) entities/ and textures/
                zf.write(root, os.path.join(prefix, rel).replace("\\", "/") + "/")
            for fn in sorted(files):
                p = os.path.join(root, fn)
                zf.write(p, os.path.join(prefix, os.path.relpath(p, src)).replace("\\", "/"))


def build(export_dir, m7_dir, out_dir, tag, name, auto=False):
    """auto: the pack installs itself when a world with it is opened (auto_functions); its own UUID and
    file names (<tag>_auto.*), so it never replaces the manual pack in a game"""
    uid = f"mct-{tag}" + ("-auto" if auto else "")
    stem = tag + ("_auto" if auto else "")
    assert re.match(r"^[a-z][a-z0-9_]*$", tag), "tag: lowercase letters, digits, _"
    meta = json.load(open(os.path.join(export_dir, "meta.json"), encoding="utf-8"))
    blocks, m7 = read_blocks(export_dir, m7_dir)
    ta = meta["tickingareas"][0]
    assert all(ta[i] <= p[i] <= ta[i + 3] for p in blocks for i in range(3)), "blocks outside the ticking area"
    chain = max(meta["chain_lengths"].values())
    chain_rule = max(65535, int(math.ceil(chain / 1e6) * 1e6))
    bp = os.path.join(out_dir, "behavior_pack")
    shutil.rmtree(bp, ignore_errors=True)
    sdir, fdir = os.path.join(bp, "structures", "mct"), os.path.join(bp, "functions", "mct")
    os.makedirs(sdir); os.makedirs(fdir)
    tiles = tiles_of(blocks)
    for i, (org, size, cells) in enumerate(tiles):
        rel = {}
        for p, (kind, facing, cond, auto, cmd) in cells.items():
            r = tuple(p[j] - org[j] for j in range(3))
            rel[r] = cb_cell(kind, facing, cond, auto, cmd, r)
        data = structure_bytes(size, rel)
        open(os.path.join(sdir, f"{tag}_{i:02d}.mcstructure"), "wb").write(data)
        print(f"  tile {i:02d} at {org} size {size}: {len(cells):,} blocks, {len(data) / 2**20:.1f} MB", flush=True)
    kb_offset = (0, 0, 0)
    if m7:
        cells, lo = keyboard_cells(m7)
        kb_offset = (5 + lo[0], lo[1], 3 + lo[2])
        size = [max(p[i] for p in cells) + 1 for i in range(3)]
        open(os.path.join(sdir, f"{tag}_kb.mcstructure"), "wb").write(structure_bytes(size, cells))
    info = {"tag": tag, "auto": auto, "model_hash": meta["model_hash"], "config": meta["config"], "params": meta["params"],
            "blocks": len(blocks), "tiles": [{"origin": o, "size": s, "blocks": len(c)} for o, s, c in tiles],
            "ticking_area": ta, "gamerule_maxcommandchainlength": chain_rule, "keyboard_offset": kb_offset,
            "init_power": m7["init_power"] if m7 else None, "pack_uuid": _uuid(uid + "-bp"),
            "pack_version": PACK_VERSION, "label": f"{meta['config'].upper()}"}
    fs = functions(tag, tiles, meta, chain_rule, kb_offset, len(blocks))
    if not m7:
        del fs["keyboard"]
    if auto:
        fs.update(auto_functions(info))
    for fn, cmds in fs.items():
        open(os.path.join(fdir, fn + ".mcfunction"), "w", encoding="utf-8", newline="\n").write("\n".join(cmds) + "\n")
    json.dump({"values": ["mct/tick"]}, open(os.path.join(bp, "functions", "tick.json"), "w"))
    man = {"format_version": 2,
           "header": {"name": name, "description": f"{meta['params']:,} 参数三值 Transformer（原版命令方块 + 记分板），{len(blocks):,} 个方块；"
                                                    + ("挂在新建的超平坦世界上，进入后自动安装（只装一次）" if auto else
                                                       "新建超平坦世界后输入 /function mct/install"),
                      "uuid": _uuid(uid + "-bp"), "version": PACK_VERSION, "min_engine_version": [1, 20, 0]},
           "modules": [{"type": "data", "uuid": _uuid(uid + "-bp-data"), "version": PACK_VERSION}]}
    for fn in ("manifest.json", "pack_manifest.json"):
        json.dump(man, open(os.path.join(bp, fn), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(info, open(os.path.join(out_dir, "pack_info.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    zip_dir(bp, os.path.join(out_dir, f"{stem}.mcpack"))
    # NetEase MC Studio component: root = behavior_pack_* + resource_pack_* (an empty resource pack is required)
    ne = os.path.join(out_dir, "_netease")
    shutil.rmtree(ne, ignore_errors=True)
    shutil.copytree(bp, os.path.join(ne, f"behavior_pack_{stem}"))
    os.makedirs(os.path.join(ne, f"behavior_pack_{stem}", "entities"))
    rp = os.path.join(ne, f"resource_pack_{stem}")
    os.makedirs(os.path.join(rp, "textures"))
    rman = {"format_version": 2,
            "header": {"name": name + " RP", "description": "占位资源包（网易组件要求）", "uuid": _uuid(uid + "-rp"),
                       "version": PACK_VERSION, "min_engine_version": [1, 20, 0]},
            "modules": [{"type": "resources", "uuid": _uuid(uid + "-rp-res"), "version": PACK_VERSION}]}
    for fn in ("manifest.json", "pack_manifest.json"):
        json.dump(rman, open(os.path.join(rp, fn), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    zip_dir(ne, os.path.join(out_dir, f"{stem}_netease.zip"))
    shutil.rmtree(ne)
    print(f"{stem}: {len(blocks):,} blocks in {len(tiles)} tiles, gamerule maxcommandchainlength {chain_rule}, "
          f"keyboard offset {kb_offset}; {out_dir}")


def verify(out_dir, export_dir, m7_dir=None):
    """read every structure back (generic NBT reader) and compare with the export, block by block"""
    info = json.load(open(os.path.join(out_dir, "pack_info.json"), encoding="utf-8"))
    tag = info["tag"]
    blocks, m7 = read_blocks(export_dir, m7_dir)
    sdir = os.path.join(out_dir, "behavior_pack", "structures", "mct")
    seen = 0
    for i, t in enumerate(info["tiles"]):
        root = loads(open(os.path.join(sdir, f"{tag}_{i:02d}.mcstructure"), "rb").read())
        sx, sy, sz = root["size"]
        st = root["structure"]
        idx = st["block_indices"][0]
        pal = st["palette"]["default"]["block_palette"]
        pos = st["palette"]["default"]["block_position_data"]
        assert all(v == -1 for v in st["block_indices"][1])
        for j, v in enumerate(idx):
            if v < 0:
                continue
            x, r = divmod(j, sy * sz); y, z = divmod(r, sz)
            p = (t["origin"][0] + x, t["origin"][1] + y, t["origin"][2] + z)
            kind, facing, cond, auto, cmd = blocks[p]
            e = pos[str(j)]["block_entity_data"]
            ok = (pal[v]["name"] == NAME[kind] and pal[v]["states"]["facing_direction"] == facing
                  and pal[v]["states"]["conditional_bit"] == int(cond) and e["Command"] == cmd
                  and e["LPCommandMode"] == MODE[kind] and e["LPCondionalMode"] == int(cond) and e["conditionalMode"] == int(cond)
                  and e["auto"] == int(auto) and e["LPRedstoneMode"] == (0 if auto else 1)
                  and (e["x"], e["y"], e["z"]) == (x, y, z) and e["Version"] == CB_VERSION)
            assert ok, (p, blocks[p], pal[v], e)
            seen += 1
        print(f"  tile {i:02d} ok ({seen:,} so far)", flush=True)
    assert seen == len(blocks), (seen, len(blocks))
    if m7:
        root = loads(open(os.path.join(sdir, f"{tag}_kb.mcstructure"), "rb").read())
        n = sum(1 for v in root["structure"]["block_indices"][0] if v >= 0)
        assert n == len(m7["keyboard"]), (n, len(m7["keyboard"]))
    print(f"PACK VERIFY PASS: {seen:,} blocks match the export" + (f", keyboard {len(m7['keyboard'])} blocks" if m7 else ""))
    return True


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda k, d: (a.pop(a.index(k) + 1), a.remove(k))[0] if k in a else d
    tag = opt("--tag", "mct")
    name = opt("--name", "MCBE Transformer")
    if "--verify" in a:
        a.remove("--verify")
        verify(a[0], a[1], a[2] if len(a) > 2 else None)
    else:
        auto = "--auto" in a
        a = [x for x in a if x != "--auto"]
        if len(a) == 3:
            build(a[0], a[1], a[2], tag, name, auto)
        else:
            build(a[0], None, a[1], tag, name, auto)
