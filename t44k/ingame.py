# -*- coding: utf-8 -*-
# T44K in-game builder + acceptance runner. Executed INSIDE the game through mod-debug-bridge
# (execute_code, server side, Python 2.7 / ModSDK). Dev-time only; the built world runs without it.
#
# Usage (paste into execute_code, then call the functions):
#   1. python -m t44k.export_plan <ASCII dir>   (offline; writes plan.json + programs.json)
#   2. set DIR below, run this file, then:  setup(); probe_names(); clear(); place_all(); init()
#   3. step_group('G0') ... to debug one group, or start_auto() + AQ to run every test prompt.
# Notes (all measured in-game):
#   - SetCommand needs a byte str: json gives unicode -> .encode('utf-8'), or every call fails.
#   - Impulse heads fire on a LATER tick, not inside the /setblock that powers them.
#   - Clear the core region before re-placing: a stale block after a new tail would be executed.
import mod.server.extraServerApi as serverApi
import time, json

DIR = 'C:\\path\\to\\export\\'          # ASCII path with plan.json / programs.json
_lv = serverApi.GetLevelId()
_cf = serverApi.GetEngineCompFactory()
_cmd = _cf.CreateCommand(_lv)
_pid = serverApi.GetPlayerList()[0]
_bi = _cf.CreateBlockInfo(_lv)
_be = _cf.CreateBlockEntity(_lv)
_game = _cf.CreateGame(_lv)
MODE = {'impulse': 0, 'repeat': 1, 'chain': 2}
NAME = {'impulse': 'minecraft:command_block', 'repeat': 'minecraft:repeating_command_block',
        'chain': 'minecraft:chain_command_block'}


def mc(c):
    return _cmd.SetCommand(c, _pid)


def rd(name, obj='t4'):
    """read a score by binary search over 'execute if score ... matches lo..hi'"""
    if not mc("/execute if score %s %s matches ..2147483647" % (name, obj)):
        return None
    lo, hi = -2 ** 31, 2 ** 31 - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if mc("/execute if score %s %s matches %d..%d" % (name, obj, lo, mid)): hi = mid
        else: lo = mid + 1
    return lo


def load():
    global PLAN, PROG, P
    PLAN = json.load(open(DIR + 'plan.json'))
    PROG = json.load(open(DIR + 'programs.json'))
    P = PLAN['power']


def setup():
    ta = PLAN['tickingarea']
    return [mc('/gamerule commandblockoutput false'), mc('/tickingarea add %d %d %d %d %d %d t44k' % tuple(ta))]


def probe_names():
    """NetEase silently drops commands whose score-holder names it filters: probe them all"""
    return [n for n in PROG['names'] if not mc('/scoreboard players set %s t4 0' % n.encode('utf-8'))]


def clear():
    ta = PLAN['tickingarea']
    lo, hi = PLAN['y_range']
    for y in range(lo, hi + 8):
        mc('/fill %d %d %d %d %d %d air' % (ta[0], y, ta[2], ta[3], y, ta[5]))


def place_all():
    for b in PLAN['blocks']:
        p = tuple(b['pos'])
        aux = b['facing'] | (8 if b['cond'] else 0)
        _bi.SetBlockNew(p, {'name': NAME[b['kind']], 'aux': aux}, 0, 0, True, False)
        _be.SetCommandBlock(p, 0, b['cmd'].encode('utf-8'), '', MODE[b['kind']], 1 if b['cond'] else 0,
                            0 if b['auto'] else 1)
    return len(PLAN['blocks'])


def init():
    return [c for c in PROG['init'] if not mc('/' + c.encode('utf-8'))]   # objectives-add may fail if present


def start(test_index, manual=False):
    for c in PROG['tests'][test_index]['start']:
        mc('/' + c.encode('utf-8'))
    if manual:                      # disarm: groups are then powered by hand with step_group()
        mc('/scoreboard players set arm t4 0')
        mc('/scoreboard players set run t4 0')


def step_group(g):
    """power one group by hand (it fires on a later tick)"""
    return mc('/setblock %d %d %d redstone_block' % tuple(P[g]))


AQ = {'i': 0, 'state': 'idle', 'res': [], 'timer': None}


def _auto():
    if AQ['state'] == 'done':
        return
    if AQ['state'] == 'idle':
        if AQ['i'] >= len(PROG['tests']):
            AQ['state'] = 'done'
            return
        start(AQ['i'])
        AQ['t0'] = time.clock(); AQ['tk0'] = rd('tk'); AQ['state'] = 'running'
        return
    if mc('/execute if score run t4 matches 0'):
        T = PROG['tests'][AQ['i']]
        got = [x for x in [rd('out%d' % p) for p in range(16)] if x != -1]
        fin = sum(1 for k, v in T['final'].items() if rd(k.encode('utf-8')) != v)
        AQ['res'].append({'i': AQ['i'], 'ok': got == T['out'] and fin == 0, 'fw': rd('fw'),
                          'wall': round(time.clock() - AQ['t0'], 2), 'ticks': rd('tk') - AQ['tk0'], 'final_bad': fin})
        AQ['i'] += 1
        AQ['state'] = 'idle'


def start_auto():
    AQ.update({'i': 0, 'state': 'idle', 'res': []})
    AQ['timer'] = _game.AddRepeatedTimer(0.05, _auto)
