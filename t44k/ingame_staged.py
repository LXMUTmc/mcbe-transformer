# -*- coding: utf-8 -*-
# Staged build (T1M ...) in-game builder + acceptance runner. Executed INSIDE the game through
# mod-debug-bridge (server side, Python 2.7 / ModSDK). Dev-time only.
#   1. python -m t44k.export_staged <ASCII dir> t1m        (offline)
#   2. set DIR, run this file, then:
#        setup(); probe_names()            -> [] expected
#        clear(); place_all()              -> repeat until it returns 'done' (batches)
#        init()                            -> [] expected (objectives-add may fail when present)
#        start_auto()                      -> runs every test prompt; read AQ['res']
import mod.server.extraServerApi as serverApi
import time, json

DIR = 'C:\\path\\to\\export\\'
_lv = serverApi.GetLevelId()
_cf = serverApi.GetEngineCompFactory()
_cmd = _cf.CreateCommand(_lv)
_bi = _cf.CreateBlockInfo(_lv)
_be = _cf.CreateBlockEntity(_lv)
_game = _cf.CreateGame(_lv)
MODE = {'impulse': 0, 'repeat': 1, 'chain': 2}
NAME = {'impulse': 'minecraft:command_block', 'repeat': 'minecraft:repeating_command_block',
        'chain': 'minecraft:chain_command_block'}
META = json.load(open(DIR + 'meta.json'))
PROG = json.load(open(DIR + 'programs.json'))
PL = {'f': None, 'n': 0}


def mc(c):
    if isinstance(c, unicode):
        c = c.encode('utf-8')
    return _cmd.SetCommand(c)


def rd(name, obj='t4'):
    if not mc('/execute if score %s %s matches ..2147483647' % (name, obj)):
        return None
    lo, hi = -2 ** 31, 2 ** 31 - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if mc('/execute if score %s %s matches %d..%d' % (name, obj, lo, mid)): hi = mid
        else: lo = mid + 1
    return lo


def setup():
    r = [mc('/gamerule commandblockoutput false'), mc('/gamerule maxcommandchainlength 65535')]
    for k, a in enumerate(META['tickingareas']):
        r.append(mc('/tickingarea add %d %d %d %d %d %d stg%s' % (a[0], a[1], a[2], a[3], a[4], a[5], 'abcdefghij'[k])))
    return r


def probe_names():
    return [n for n in PROG['names'] if not mc('/scoreboard players set %s t4 0' % n.encode('utf-8'))]


def clear(y_lo=-60, y_hi=40):
    for a in META['tickingareas']:
        for y in range(y_lo, y_hi + 1):
            mc('/fill %d %d %d %d %d %d air' % (a[0], y, a[2], a[3], y, a[5]))


def place_all(batch=150000):
    """place the next batch of blocks from blocks.txt; call again until it returns 'done'"""
    if PL['f'] is None:
        PL['f'] = open(DIR + 'blocks.txt'); PL['n'] = 0
    f, k, t0 = PL['f'], 0, time.clock()
    while k < batch:
        line = f.readline()
        if not line:
            f.close(); PL['f'] = None
            return 'done', PL['n'], round(time.clock() - t0, 1)
        head, cmd = line.rstrip('\n').split('\t', 1)
        x, y, z, kind, facing, cond, auto = head.split(' ')
        p = (int(x), int(y), int(z))
        cond, auto = cond == 'True' or cond == '1', auto == 'True' or auto == '1'
        _bi.SetBlockNew(p, {'name': NAME[kind], 'aux': int(facing) | (8 if cond else 0)}, 0, 0, True, False)
        _be.SetCommandBlock(p, 0, cmd, '', MODE[kind], 1 if cond else 0, 0 if auto else 1)
        k += 1; PL['n'] += 1
    return 'more', PL['n'], round(time.clock() - t0, 1)


def init():
    return [c for c in PROG['init'] if not mc('/' + c.encode('utf-8'))]


AQ = {'i': 0, 'state': 'idle', 'res': [], 'timer': None}


def _auto():
    if AQ['state'] == 'done':
        return
    if AQ['state'] == 'idle':
        if AQ['i'] >= len(PROG['tests']):
            AQ['state'] = 'done'
            return
        AQ['tk0'] = rd('tk'); AQ['t0'] = time.clock()
        for c in PROG['tests'][AQ['i']]['start']:
            mc('/' + c.encode('utf-8'))
        AQ['state'] = 'running'
        return
    if mc('/execute if score run t4 matches 0'):
        T = PROG['tests'][AQ['i']]
        wall = time.clock() - AQ['t0']
        got = [x for x in [rd('out%d' % p) for p in range(META['T'])] if x != -1]
        fin = sum(1 for k, v in T['final'].items() if rd(k.encode('utf-8')) != v)
        fw = rd('fw')
        AQ['res'].append({'i': AQ['i'], 'ok': got == T['out'] and fin == 0, 'got': got, 'want': T['out'],
                          'final_bad': fin, 'fw': fw, 'wall_s': round(wall, 2), 'ticks': rd('tk') - AQ['tk0'],
                          's_per_forward': round(wall / max(1, fw), 3)})
        AQ['i'] += 1
        AQ['state'] = 'idle'


def start_auto():
    AQ.update({'i': 0, 'state': 'idle', 'res': []})
    AQ['timer'] = _game.AddRepeatedTimer(0.5, _auto)
