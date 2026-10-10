# -*- coding: utf-8 -*-
# M6 keyboard wall builder. Executed INSIDE the game through mod-debug-bridge (server side,
# Python 2.7 / ModSDK). Dev-time only; the finished wall is plain vanilla blocks.
#   1. python -m t44k.m6 <ASCII dir>          (offline; m6.json now carries "keyboard")
#   2. set M6 below, run this file, then: clear_wall(ox, oy, zb); place_wall(ox, oy, zb)
# (ox, oy, zb) = keyboard origin: key (x, y) sits at (ox + x, oy + y), the stone plane is z = zb.
# The player stands north of the wall (z < zb - 1) and faces south.
import mod.server.extraServerApi as serverApi
import json

M6 = 'C:\\path\\to\\m6.json'
_lv = serverApi.GetLevelId()
_cf = serverApi.GetEngineCompFactory()
_cmd = _cf.CreateCommand(_lv)
_bi = _cf.CreateBlockInfo(_lv)
_be = _cf.CreateBlockEntity(_lv)
M = json.load(open(M6))
KB = M['keyboard']


def clear_wall(ox, oy, zb, pad=3):
    xs = [ox + b['rel'][0] for b in KB]
    ys = [oy + b['rel'][1] for b in KB]
    for z in range(zb - 1 - pad, zb + max(b['rel'][2] for b in KB) + 1 + pad):   # the old wall may sit on other planes
        _cmd.SetCommand('/fill %d %d %d %d %d %d air' % (min(xs) - pad, min(ys), z, max(xs) + pad, max(ys) + pad, z))


def place_wall(ox, oy, zb):
    """solid blocks first (buttons and signs need something to hang on); returns the signs that failed"""
    hang = ('minecraft:stone_button', 'minecraft:wall_sign')
    order = [b for b in KB if b['name'] not in hang] + [b for b in KB if b['name'] in hang]
    bad = []
    for b in order:
        p = (ox + b['rel'][0], oy + b['rel'][1], zb + b['rel'][2])
        _bi.SetBlockNew(p, {'name': b['name'], 'aux': b['aux']}, 0, 0, True, False)
        if 'cmd' in b:
            _be.SetCommandBlock(p, 0, b['cmd'].encode('utf-8'), '', b['mode'], 0, 0 if b['auto'] else 1)
        if 'text' in b and not _bi.SetSignBlockText(p, b['text'].encode('utf-8'), 0):
            bad.append(b['text'])
    return bad
