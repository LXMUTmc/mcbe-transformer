# -*- coding: utf-8 -*-
# Prefix bisection of a placed single-chain build (Python 2, inside the game via mod-debug-bridge).
# Needs the globals of t44k/ingame_staged.py in G (mc, rd, _bi, _be, NAME, MODE, DIR).
# Runs by itself on a repeated timer, so the player only has to keep the game in the foreground:
#   idle -> for each cut k: cut the chain at k (air), power the head from below with run == 0
#   (control inactive: every tick recomputes identical values), record N ticks, unpower, restore.
# Results: BIS['res'] = [(label, [tick durations in ms])]; BIS['state'] == 'done' at the end.
import time

BIS = {'state': 'init', 'res': [], 'log': [], 'timer': None}
CUTS = [100000, 300000, 600000, None]       # None = the full chain
TICKS_PER_PHASE = 4
POWER = (32768, 99, 1)                      # below the repeating head at (32768, 100, 1)


def _orig(ks):
    out = {}
    with open(G['DIR'] + 'blocks.txt') as f:
        for i, line in enumerate(f):
            if i in ks:
                head, cmd = line.rstrip('\n').split('\t', 1)
                out[i] = (head.split(' '), cmd)
            if i > max(ks):
                break
    return out


ORIG = _orig(set(k for k in CUTS if k))


def _cut(k):
    h = ORIG[k][0]
    G['mc']('/setblock %s %s %s air' % (h[0], h[1], h[2]))


def _restore(k):
    h, c = ORIG[k]
    p = (int(h[0]), int(h[1]), int(h[2]))
    cond, auto = h[5] in ('True', '1'), h[6] in ('True', '1')
    G['_bi'].SetBlockNew(p, {'name': G['NAME'][h[3]], 'aux': int(h[4]) | (8 if cond else 0)}, 0, 0, True, False)
    G['_be'].SetCommandBlock(p, 0, c, '', G['MODE'][h[3]], 1 if cond else 0, 0 if auto else 1)


def _power(on):
    G['mc']('/setblock %d %d %d %s' % (POWER + ('redstone_block' if on else 'air',)))


def _tick():
    """called every server tick (possibly more than once per tick: use the tk counter)"""
    mc = G['mc']
    k = BIS.get('last_tk', 0)
    if not mc('/execute if score tk t4 matches %d..' % (k + 1)):
        return
    while mc('/execute if score tk t4 matches %d..' % (k + 1)):
        k += 1
    now = time.time()
    if BIS.get('last_t') is not None:
        BIS['cur'].append(int((now - BIS['last_t']) * 1000))
    BIS['last_t'], BIS['last_tk'] = now, k
    phase = BIS['phase']
    need = 20 if phase == 'idle' else TICKS_PER_PHASE + 1      # +1: the first tick after powering is partial
    if len(BIS['cur']) < need:
        return
    BIS['res'].append((phase, list(BIS['cur'])))
    # leave the current phase
    if phase != 'idle':
        _power(False)
        if BIS['k'] is not None:
            _restore(BIS['k'])
    if not BIS['todo']:
        BIS['state'] = 'done'
        G['_game'].CancelTimer(BIS['timer'])
        return
    BIS['k'] = BIS['todo'].pop(0)
    BIS['phase'] = 'prefix %s' % (BIS['k'] if BIS['k'] else 'full')
    if BIS['k'] is not None:
        _cut(BIS['k'])
    _power(True)
    BIS['cur'], BIS['last_t'] = [], None


def start():
    G['mc']('/scoreboard players set run t4 0')
    BIS.update({'state': 'running', 'res': [], 'phase': 'idle', 'k': None, 'todo': list(CUTS),
                'cur': [], 'last_t': None, 'last_tk': G['rd']('tk')})
    BIS['timer'] = G['_game'].AddRepeatedTimer(0.05, _tick)   # never 0.0: it can fire many times per busy tick
