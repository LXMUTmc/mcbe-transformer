"""Build a ready-to-play .mcworld from a behavior pack made by t44k/pack.py: a NEW flat world that installs
the model by itself the first time it is opened.

The world is generated from the template t44k/world_template/level.dat (an empty flat creative world,
cheats and command blocks on, made by the game) with an empty db/ (the game creates the terrain).
The behavior pack is embedded in the world (behavior_packs/ + world_behavior_packs.json for vanilla
Bedrock, netease_world_behavior_packs.json for NetEase) with the auto-install functions of
t44k/pack.auto_functions (install once on first open, keyboard in front of the player, 初始化 pressed).
NetEase MC Studio imports such a folder as an add-on, not a map: on NetEase use the auto pack
(python -m t44k.pack ... --auto) as a component on a new flat world instead.
Level settings changed from the template: name, spawn point (spawn radius 0), coordinates shown,
daylight / weather cycle and mob spawning off, maxcommandchainlength from the pack.

  python -m t44k.world_build PACK_DIR OUT.mcworld [--name "MCBE Transformer T10M"]
      PACK_DIR: the output folder of t44k.pack (behavior_pack/ + pack_info.json)
"""
import sys, os, json, shutil, tempfile, zipfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from t44k import leveldat as L
from t44k.pack import auto_functions

SPAWN = (0, -60, 0)            # flat world surface (bedrock, 2 dirt, grass: the player stands at y = -60)


def level_dat(dst, name, chain_rule):
    ver, r = L.read(os.path.join(HERE, "world_template", "level.dat"))

    def setv(k, v):
        r[k] = (r[k][0], v)
    setv("LevelName", name); setv("LastPlayed", int(time.time()))
    setv("SpawnX", SPAWN[0]); setv("SpawnY", SPAWN[1]); setv("SpawnZ", SPAWN[2]); setv("spawnradius", 0)
    setv("maxcommandchainlength", chain_rule); setv("commandblockoutput", 0); setv("commandblocksenabled", 1)
    setv("commandsEnabled", 1); setv("showcoordinates", 1)
    setv("dodaylightcycle", 0); setv("doweathercycle", 0); setv("domobspawning", 0)
    L.write(os.path.join(dst, "level.dat"), ver, r)


def main(pack_dir, out, name):
    info = json.load(open(os.path.join(pack_dir, "pack_info.json"), encoding="utf-8"))
    tmp = tempfile.mkdtemp(prefix="mct_newworld_")
    try:
        w = os.path.join(tmp, "world")
        os.makedirs(os.path.join(w, "db"))
        level_dat(w, name, info["gamerule_maxcommandchainlength"])
        open(os.path.join(w, "levelname.txt"), "w", encoding="utf-8").write(name)
        bp = os.path.join(w, "behavior_packs", f"mct_{info['tag']}")
        shutil.copytree(os.path.join(pack_dir, "behavior_pack"), bp)
        for fn, cmds in auto_functions(info).items():
            open(os.path.join(bp, "functions", "mct", fn + ".mcfunction"), "w", encoding="utf-8",
                 newline="\n").write("\n".join(cmds) + "\n")
        ref = [{"pack_id": info["pack_uuid"], "version": info["pack_version"]}]
        for fn in ("world_behavior_packs.json", "netease_world_behavior_packs.json"):
            json.dump(ref, open(os.path.join(w, fn), "w"), indent=1)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for root, dirs, files in os.walk(w):
                rel = os.path.relpath(root, w)
                if rel != ".":
                    zf.write(root, rel.replace("\\", "/") + "/")
                for fn in sorted(files):
                    p = os.path.join(root, fn)
                    zf.write(p, os.path.relpath(p, w).replace("\\", "/"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{out}: {os.path.getsize(out) / 2**20:.1f} MB, '{name}', auto-install {len(info['tiles'])} tiles, "
          f"keyboard at spawn {SPAWN}")


if __name__ == "__main__":
    a = sys.argv[1:]
    name = "MCBE Transformer"
    if "--name" in a:
        i = a.index("--name"); name = a[i + 1]; del a[i:i + 2]
    main(a[0], a[1], name)
