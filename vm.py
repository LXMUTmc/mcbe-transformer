"""A strict interpreter for the subset of Bedrock commands the compiler emits.
It executes the literal command TEXT, so it verifies the compiler, not just the math.

Strictness: reading an unset score, dividing by <=0, a negative dividend in /= or %=,
or leaving the int32 range are all recorded as faults.
"""
import json, re

INT_MIN, INT_MAX = -2 ** 31, 2 ** 31 - 1


class VM:
    def __init__(self, div_mode="trunc"):
        self.s = {}                 # (name, obj) -> int
        self.objs = set()
        self.out = []               # tellraw texts
        self.faults = []
        self.executed = 0           # commands looked at
        self.effective = 0          # commands that actually changed / did something
        self.div_mode = div_mode
        self.functions = {}
        self.blocks = {}

    # ------------------------------------------------------------------
    def get(self, n, o):
        if (n, o) not in self.s:
            self.faults.append(f"unset read {n} {o}")
            return None
        return self.s[(n, o)]

    def put(self, n, o, v):
        if not INT_MIN <= v <= INT_MAX:
            self.faults.append(f"overflow {n} {o} = {v}")
            v = (v + 2 ** 31) % 2 ** 32 - 2 ** 31
        self.s[(n, o)] = v

    @staticmethod
    def in_range(v, r):
        if ".." in r:
            lo, hi = r.split("..")
            return (lo == "" or v >= int(lo)) and (hi == "" or v <= int(hi))
        return v == int(r)

    def div(self, a, b):
        if b == 0:
            self.faults.append("div by zero"); return a
        if a < 0 or b < 0:
            self.faults.append(f"negative division {a}/{b}")
        q = abs(a) // abs(b)
        if self.div_mode == "floor":
            return a // b
        return q if (a >= 0) == (b >= 0) else -q

    # ------------------------------------------------------------------
    def run(self, cmd):
        self.executed += 1
        t = cmd.split(" ")
        if t[0] == "execute":
            i = 1
            while i < len(t) and t[i] in ("if", "unless"):
                neg = t[i] == "unless"
                assert t[i + 1] == "score"
                a = self.s.get((t[i + 2], t[i + 3]))
                if t[i + 4] == "matches":
                    ok = a is not None and self.in_range(a, t[i + 5]); i += 6
                else:
                    b = self.s.get((t[i + 5], t[i + 6])); cmpop = t[i + 4]
                    ok = a is not None and b is not None and {
                        "<": a < b, "<=": a <= b, "=": a == b, ">=": a >= b, ">": a > b}[cmpop]
                    i += 7
                if ok == neg:
                    return False
            if i == len(t):                 # bare condition ("gate"): succeeds when all tests pass
                return True
            assert t[i] == "run", cmd
            return self.run(" ".join(t[i + 1:])) or True
        self.effective += 1
        if t[0] == "scoreboard":
            if t[1] == "objectives":
                self.objs.add(t[3]); return True
            n, o = t[3], t[4]
            assert o in self.objs, f"unknown objective {o}"
            if t[2] == "set":
                self.put(n, o, int(t[5]))
            elif t[2] == "add":
                v = self.get(n, o)
                if v is not None: self.put(n, o, v + int(t[5]))
            elif t[2] == "remove":
                v = self.get(n, o)
                if v is not None: self.put(n, o, v - int(t[5]))
            elif t[2] == "operation":
                opx, bn, bo = t[5], t[6], t[7]
                b = self.get(bn, bo)
                if b is None: return False
                if opx == "=":
                    self.put(n, o, b); return True
                a = self.get(n, o)
                if a is None: return False
                if opx == "+=": r = a + b
                elif opx == "-=": r = a - b
                elif opx == "*=": r = a * b
                elif opx == "/=": r = self.div(a, b)
                elif opx == "%=": r = a - b * self.div(a, b)
                elif opx == ">": r = max(a, b)
                elif opx == "<": r = min(a, b)
                else: raise ValueError(cmd)
                self.put(n, o, r)
            else:
                raise ValueError(cmd)
            return True
        if t[0] == "tellraw":
            js = json.loads(cmd.split(" ", 2)[2])
            txt = ""
            for part in js["rawtext"]:
                if "text" in part: txt += part["text"]
                elif "score" in part: txt += str(self.s.get((part["score"]["name"], part["score"]["objective"]), "?"))
            self.out.append(re.sub("§.", "", txt)); return True
        if t[0] == "function":
            for c in self.functions[t[1]]:
                self.run(c)
            return True
        if t[0] == "setblock":
            self.blocks[tuple(t[1:4])] = t[4]; return True
        if t[0] == "say":
            self.out.append(" ".join(t[1:])); return True
        raise ValueError("unsupported: " + cmd)

    def run_all(self, cmds):
        for c in cmds:
            self.run(c)
