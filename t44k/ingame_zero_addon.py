# -*- coding: utf-8 -*-
# Zero-add-on speed test rig (Python 2, inside the game via mod-debug-bridge; dev-time only).
# Builds, next to the player, pure vanilla controls for an already placed single-chain T1M:
#   - a stone button that runs an impulse chain: the start program of test prompt IDX (sets the prompt,
#     out*, pos, cur, fw, run) and finally places the redstone block next to the chain head
#   - a tick counter: repeating command block (needs redstone) + redstone block: 'add probe t4 1'
# The chain removes its own power when generation ends, so the world is saved unpowered.
# The player can read progress without add-ons: /scoreboard players list fw   /scoreboard players list probe
import mod.server.extraServerApi as serverApi
import json

DIR = 'C:\\path\\to\\export\\'           # the single-chain export (meta.json, programs.json)
IDX = 2                                   # test prompt (2 = 中华)
_lv = serverApi.GetLevelId()
_cf = serverApi.GetEngineCompFactory()
_bi = _cf.CreateBlockInfo(_lv)
_be = _cf.CreateBlockEntity(_lv)
META = json.load(open(DIR + 'meta.json'))
PROG = json.load(open(DIR + 'programs.json'))


def _cb(p, name, aux, cmd, mode, redstone):
    _bi.SetBlockNew(p, {'name': 'minecraft:air', 'aux': 0}, 0, 0, True, True)      # fresh block entity
    _bi.SetBlockNew(p, {'name': name, 'aux': aux}, 0, 0, True, True)
    _be.SetCommandBlock(p, 0, cmd.encode('utf-8') if isinstance(cmd, unicode) else cmd, '', mode, 0, redstone)


def build(ox, oy, oz):
    """button at (ox, oy, oz-1) on a stone at (ox, oy, oz); impulse chain from (ox, oy, oz+1) going +z;
    tick counter at (ox+3, oy, oz+1) powered by a redstone block at (ox+3, oy, oz)"""
    T = PROG['tests'][IDX]
    cmds = [c for c in T['start'] if not c.startswith('setblock')]
    P = META['power']['G']
    cmds += ['setblock %d %d %d air' % tuple(P), 'setblock %d %d %d redstone_block' % tuple(P),
             'tellraw @a {"rawtext":[{"text":"\\u00a7a[T1M] start: %s"}]}' % T['text'].split(u'\u2192')[0].encode('unicode_escape')]
    _bi.SetBlockNew((ox, oy, oz), {'name': 'minecraft:stone', 'aux': 0}, 0, 0, True, True)
    for i, c in enumerate(cmds):
        p = (ox, oy, oz + 1 + i)
        if i == 0:
            _cb(p, 'minecraft:command_block', 3, c, 0, 1)                 # impulse, needs redstone, facing +z
        else:
            _cb(p, 'minecraft:chain_command_block', 3, c, 2, 0)           # chain, always active
    _bi.SetBlockNew((ox, oy, oz - 1), {'name': 'minecraft:stone_button', 'aux': 2}, 0, 0, True, True)
    _cb((ox + 3, oy, oz + 1), 'minecraft:repeating_command_block', 3, 'scoreboard players add probe t4 1', 1, 1)
    _bi.SetBlockNew((ox + 3, oy, oz), {'name': 'minecraft:redstone_block', 'aux': 0}, 0, 0, True, True)
    return len(cmds)
