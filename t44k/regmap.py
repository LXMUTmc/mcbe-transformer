"""Register remapping: few score-holder NAMES, many objectives.

Measured in-game (2026-10-01, 50K-command chains, one tick each):
  writes rotating over   1,000 / 5,000 / 20,000 different fake-player names in ONE objective
                           0.20  /  0.23 / 34.8 s per tick    (same-name baseline 0.18 s)
  20,000 different names spread over 100 objectives               10.3 s
  20,000 (name, objective) pairs made of 200 names x 100 objectives  0.22 s
So the cost explodes with the number of DIFFERENT score holders (fake players) whose scores change
in one tick, not with the number of scores or objectives. T1M changes ~60K different holders per
forward, which is the 'single-tick superlinearity' we chased for days.

The fix: every bulk register NAME becomes a pair (holder, objective) with only H holder names
(q0..q{H-1}) reused across many objectives (v0, v1, ...). Control registers (run, pos, cur, fw,
t*, out*, c* snapshots, ...) keep their names in t4 so the controller, the self-test and the
in-game test tools stay unchanged. The integers computed are identical; only names change.
"""
import re

OBJ = "t4"
H = 256
KEEP = re.compile(r"^(run|pos|cur|fw|stg|live|go|plen|tk|grp|arm|gen|kk|nxt|in|st|ok|err|probe|t\d+|out\d+|c.*)$")
_REF = re.compile(r"(\S+) t4(?= |$)")


def build_map(names):
    """names: every score-holder name used with objective t4 -> {name: (holder, objective)}"""
    bulk = sorted(n for n in names if not KEEP.match(n))
    return {n: (f"q{k % H}", f"v{k // H}") for k, n in enumerate(bulk)}


def objectives(m):
    return sorted({o for _, o in m.values()}, key=lambda s: int(s[1:]))


_KW = {"scoreboard", "players", "objectives", "set", "add", "remove", "operation", "reset", "random", "list",
       "execute", "if", "unless", "score", "matches", "run", "=", "+=", "-=", "*=", "/=", "%=", ">", "<", "><"}


def _refs(cmd):
    """indices i of tokens that are score-holder names used with objective t4"""
    t = cmd.split(" ")
    return t, [i for i in range(len(t) - 1) if t[i + 1] == OBJ and t[i] not in _KW and not t[i].startswith("{")]


def remap_cmd(cmd, m):
    if cmd.lstrip("?").startswith("scoreboard objectives"):
        return cmd
    t, idx = _refs(cmd)
    for i in idx:
        if t[i] in m:
            t[i], t[i + 1] = m[t[i]]
    return " ".join(t)


def names_in(cmds):
    out = set()
    for c in cmds:
        if not c.lstrip("?").startswith("scoreboard objectives"):
            t, idx = _refs(c.lstrip("?"))
            out.update(t[i] for i in idx)
    return out


def init_objectives(m):
    return [f"scoreboard objectives add {o} dummy" for o in objectives(m)]
