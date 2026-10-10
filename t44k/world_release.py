"""Turn an installed world into a shareable .mcworld (run with the game closed or at the main menu).

A COPY of the world folder is made; the copy
  - drops every add-on reference (world_*_packs.json, NetEase's netease_world_*_packs.json, the
    behavior_packs / resource_packs folders): the model is plain vanilla blocks once installed, and
    development add-ons such as DebugBridge must not ship
  - gets a new name (levelname.txt and the LevelName string inside level.dat)
and is zipped (folder contents at the archive root) into OUT.mcworld. The original world is untouched.

  python -m t44k.world_release WORLD_DIR OUT.mcworld [--name "MCBE Transformer T10M"]
"""
import sys, os, shutil, struct, zipfile, tempfile

DROP = ("world_behavior_packs.json", "world_resource_packs.json", "netease_world_behavior_packs.json",
        "netease_world_resource_packs.json", "behavior_packs", "resource_packs", "studio.json")


def rename_level_dat(path, name):
    """level.dat = int32 storage version + int32 length + little-endian NBT; replace the LevelName string"""
    b = open(path, "rb").read()
    key = b"\x08" + struct.pack("<H", 9) + b"LevelName"
    i = b.find(key)
    assert i > 0, "LevelName not found in level.dat"
    j = i + len(key)
    n = struct.unpack_from("<H", b, j)[0]
    new = name.encode("utf-8")
    body = b[8:j] + struct.pack("<H", len(new)) + new + b[j + 2 + n:]
    open(path, "wb").write(b[:4] + struct.pack("<i", len(body)) + body)


def main(world, out, name):
    assert os.path.exists(os.path.join(world, "level.dat")), "not a world folder"
    tmp = tempfile.mkdtemp(prefix="mct_world_")
    try:
        dst = os.path.join(tmp, "world")
        shutil.copytree(world, dst)
        for d in DROP:
            p = os.path.join(dst, d)
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.exists(p):
                os.remove(p)
        for f in ("level.dat_old",):
            if os.path.exists(os.path.join(dst, f)):
                os.remove(os.path.join(dst, f))
        rename_level_dat(os.path.join(dst, "level.dat"), name)
        open(os.path.join(dst, "levelname.txt"), "w", encoding="utf-8").write(name)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for root, _, files in os.walk(dst):
                for fn in sorted(files):
                    p = os.path.join(root, fn)
                    zf.write(p, os.path.relpath(p, dst).replace("\\", "/"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{out}: {os.path.getsize(out) / 2**20:.1f} MB ('{name}')")


if __name__ == "__main__":
    a = sys.argv[1:]
    name = "MCBE Transformer"
    if "--name" in a:
        i = a.index("--name"); name = a[i + 1]; del a[i:i + 2]
    main(a[0], a[1], name)
