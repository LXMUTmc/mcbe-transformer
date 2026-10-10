# -*- coding: utf-8 -*-
# M7 builder (pinyin input world). Executed INSIDE the game through mod-debug-bridge (server side,
# Python 2.7 / ModSDK). Dev-time only; the finished world is plain vanilla blocks.
#   0. the T1M core is placed first with t44k/ingame_staged.py (same export dir: setup(), place_all())
#   1. python -m t44k.m7 <export dir> <ASCII dir>      (offline: m7.json)
#   2. set M7 below, run this file, then:
#        probe_texts()          -> [] expected (display texts NetEase would swallow)
#        place_far()            -> IME chain, self-test poller, init chain (absolute coordinates)
#        place_wall_at_player() -> keyboard 3 blocks south of the player (player faces south)
import mod.server.extraServerApi as serverApi
import json, re

M7 = 'C:\\path\\to\\m7.json'
_lv = serverApi.GetLevelId()
_cf = serverApi.GetEngineCompFactory()
_cmd = _cf.CreateCommand(_lv)
_bi = _cf.CreateBlockInfo(_lv)
_be = _cf.CreateBlockEntity(_lv)
M = json.load(open(M7))
MODE = {'impulse': 0, 'repeat': 1, 'chain': 2}
NAME = {'impulse': 'minecraft:command_block', 'repeat': 'minecraft:repeating_command_block',
        'chain': 'minecraft:chain_command_block'}


def mc(c):
    if isinstance(c, unicode):
        c = c.encode('utf-8')
    return _cmd.SetCommand(c)


def probe_texts():
    """run every display text of the far chains once as an actionbar title; returns the ones that fail"""
    bad = []
    for b in M['blocks']:
        c = b['cmd']
        m = re.search(r'(tellraw|titleraw) @a (actionbar )?(\{.*\})$', c)
        if m and not mc(u'/titleraw @a actionbar ' + m.group(3)):
            bad.append(m.group(3))
    return bad


def place_far():
    for b in M['blocks']:
        p = tuple(b['pos'])
        _bi.SetBlockNew(p, {'name': 'minecraft:air', 'aux': 0}, 0, 0, True, False)       # fresh block entity
        _bi.SetBlockNew(p, {'name': NAME[b['kind']], 'aux': b['facing'] | (8 if b['cond'] else 0)}, 0, 0, True, False)
        _be.SetCommandBlock(p, 0, b['cmd'].encode('utf-8'), '', MODE[b['kind']], 1 if b['cond'] else 0, 0 if b['auto'] else 1)
    return len(M['blocks'])


def place_wall(ox, oy, zb):
    """key (x, y) at (ox + x, oy + y); stone plane z = zb; solid blocks first, then buttons and signs"""
    KB = M['keyboard']
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


def place_wall_at_player():
    pid = serverApi.GetPlayerList()[0]
    x, y, z = [int(round(v)) for v in _cf.CreatePos(pid).GetFootPos()]
    ox, oy, zb = x + 5, y, z + 3
    return (ox, oy, zb), place_wall(ox, oy, zb)
