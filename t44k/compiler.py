"""T44K compiler: spec model -> scoreboard command programs.

One forward pass is split into TICK GROUPS (each group = one command-block chain run in one
tick, E0: <= ~40K commands/tick is safe):
  G0  EMBED + layer 0      G1  layer 1      G2  final RMSNorm + LM head + argmax + CONTROL
Registers are SSA (every layer/stage writes its own names), so re-running a group gives the
same values; only CONTROL (pos/cur/out/run) is non-idempotent and is guarded by grp == 2.

A command starting with '?' is a CONDITIONAL chain block: it runs only if the previous block
in the chain succeeded. 'gate' commands (bare 'execute if score ...') followed by conditional
blocks give cheap dispatch: embedding row / position row / KV slot select only execute the
selected segment instead of V*D guarded commands.
"""
from t44k import spec as S

OBJ, CO = "t4", "t4c"


def sp(n, v):        return f"scoreboard players set {n} {OBJ} {v}"
def add(n, v):       return f"scoreboard players add {n} {OBJ} {v}" if v >= 0 else f"scoreboard players remove {n} {OBJ} {-v}"
def op(a, o, b, bo=OBJ): return f"scoreboard players operation {a} {OBJ} {o} {b} {bo}"
def ifm(n, r, cmd):  return f"execute if score {n} {OBJ} matches {r} run {cmd}"
def gate(n, r):      return f"execute if score {n} {OBJ} matches {r}"
def cond(cmd):       return "?" + cmd


def rng(lo, hi):
    if hi is None: return f"{lo}.."
    if lo is None: return f"..{hi}"
    return str(lo) if lo == hi else f"{lo}..{hi}"


def guard(n, v, cmd):
    pre = f"execute if score {n} {OBJ} matches {v} "
    return pre + cmd[len("execute "):] if cmd.startswith("execute ") else pre + "run " + cmd


def tell(text):
    return 'tellraw @a {"rawtext":[{"text":"' + text + '"}]}'


# register names -------------------------------------------------------------
# NetEase's sensitive-word filter silently drops commands whose score-holder NAME contains
# certain digit sequences once separators are removed (measured: 'K0_7_31', 'K0_9_18',
# 'a6_4_89', 'a89p64', 'a8964' are dropped). Indices are therefore separated by LETTERS, and
# every name the compiler emits is probed in-game before a build (t44k/ingame).
def nK(l, p, i): return f"K{l}p{p}d{i}"
def nV(l, p, i): return f"V{l}p{p}d{i}"
def nS(l, h, j): return f"s{l}h{h}t{j}"             # not 'j': 'j8' is filtered
def nW(l, h, j): return f"w{l}h{h}t{j}"
def nL(prefix, l, i): return f"{prefix}{l}x{i}"          # per-layer vectors: na0x5, q1x31, f0x111 ...
def nDen(l, h): return f"den{l}h{h}"
def nOff(l, h): return f"off{l}h{h}"
def snap(n): return "c" + n                               # snapshot of the last forward
def snap_names(): return [f"xf{S.L-1}x{i}" for i in range(S.D)] + ["bt", "blg"]


# SPLIT marks a point between two self-contained units (every unit starts its registers with '='
# or 'set'), so a chain may end there: re-running either chain alone gives the same values.
# stages() packs units into chains; flatten() drops the markers.
SPLIT = "#SPLIT"


class Compiler:
    def __init__(self, q, vocab=None, staged=False, cse=False, pack=False):
        """staged=True: every SPLIT unit is self-contained (no register is modified by two units),
        so stages() may cut the forward into many chains; T44K keeps staged=False (byte-identical
        to the M6 world). cse=True (staged only): ternary matmuls share common signed pairs
        (t44k/cse.py), same integers with ~1.7x fewer matmul commands"""
        self.q = q
        self.staged = staged
        self.cse = cse and staged
        self.pack = pack and staged        # embedding / position tables: 3 int8 values per score
        self.HB = 128                      # staged LM head: tokens per argmax block
        self.ctl_guard = ("grp", 2)
        self.vocab = vocab or (["<PAD>", "<BOS>", "<EOS>", "<UNK>"] + [f"#{i}" for i in range(4, S.V)])
        self.consts = {}

    def const(self, v):
        name = f"c{v}"
        self.consts[name] = v
        return name

    # ---------------------------------------------------------------- kernels
    def rdiv_c(self, n, Dv):
        K = S.koff(Dv)
        return [add(n, K * Dv + Dv // 2), op(n, "/=", self.const(Dv), CO), add(n, -K)]

    def tern_row(self, dst, row, src):
        out, first = [], True
        for i, w in enumerate(row):
            if w == 0:
                continue
            if first:
                if w > 0: out.append(op(dst, "=", src(i)))
                else:     out += [sp(dst, 0), op(dst, "-=", src(i))]
                first = False
            else:
                out.append(op(dst, "+=" if w > 0 else "-=", src(i)))
        if first:
            out.append(sp(dst, 0))
        return out

    def mat(self, tag, W, src):
        """a ternary matrix W (list of rows) applied to src(i). Returns (prep, row):
        prep = commands computing the CSE temporaries (one unit each), row(j, dst) = commands for
        output j. Without CSE: no prep, row = tern_row."""
        if not self.cse:
            return [], (lambda j, dst: self.tern_row(dst, W[j], src))
        from t44k import cse
        temps, rows = cse.plan(W)
        n = len(W[0])
        name = lambda k: f"u{tag}y{k}"
        col = lambda c: src(c) if c < n else name(c - n)
        prep = []
        for k, (i, rel, j) in enumerate(temps):
            prep += [op(name(k), "=", col(i)), op(name(k), "+=" if rel > 0 else "-=", col(j)), SPLIT]

        def row(j, dst):
            terms = sorted(rows[j].items(), key=lambda cs: (cs[1] < 0, cs[0]))    # a positive term first
            if not terms:
                return [sp(dst, 0)]
            c0, s0 = terms[0]
            out = [op(dst, "=", col(c0))] if s0 > 0 else [sp(dst, 0), op(dst, "-=", col(c0))]
            return out + [op(dst, "+=" if s > 0 else "-=", col(c)) for c, s in terms[1:]]
        return prep, row

    PB, PO = 255, 127                      # packed tables: base and offset (values in [-127, 127])

    @staticmethod
    def safe_set(name, v):
        """'set name v', or 'set name v-y' + 'add name y' when the number would contain a NetEase
        filtered digit sequence (packed constants such as 9896429 do: the command is silently dropped)"""
        import netease_lint
        bad = lambda cmd: netease_lint._pat.search(cmd)
        if not bad(sp(name, v)):
            return [sp(name, v)]
        for y in range(1, 1000):
            if not bad(sp(name, v - y)) and not bad(add(name, y)):
                return [sp(name, v - y), add(name, y)]
        raise ValueError(f"no filter-safe split for {v}")

    def packed_table(self, tag, gate_name, keys, table, dst):
        """table[key] = D integers in [-127, 127]. Instead of one conditional 'set' per element and
        key, every key sets ceil(D/3) packed scores sum((e + 127) * 255^k) (< 255^3 < 2^31); one
        decode unit shared by all keys unpacks them into dst(i). All operands are non-negative, so
        the result does not depend on the platform's division semantics."""
        groups = [list(range(i, min(i + 3, S.D))) for i in range(0, S.D, 3)]
        pk = lambda g: f"{tag}k{g}"
        c = []
        for key in keys:
            row = table[key]
            assert all(-self.PO <= v <= self.PO for v in row), f"{tag}: value outside [-127, 127]"
            c.append(gate(gate_name, key))
            for g, grp in enumerate(groups):
                c += [cond(x) for x in self.safe_set(pk(g), sum((row[i] + self.PO) * self.PB ** k for k, i in enumerate(grp)))]
            c.append(SPLIT)
        for g, grp in enumerate(groups):
            for k, i in enumerate(grp):
                c.append(op(dst(i), "=", pk(g)))
                if k:
                    c.append(op(dst(i), "/=", self.const(self.PB ** k), CO))
                if k < len(grp) - 1:
                    c.append(op(dst(i), "%=", self.const(self.PB), CO))
                c.append(add(dst(i), -self.PO))
        return c + [SPLIT]

    def rms(self, src, dst, which):
        lut = self.q["rms_luts"][which]
        c = []
        for i in range(S.D):
            c += [op("xs", "=", src(i))] + self.rdiv_c("xs", S.PS) + [op("xs", "*=", "xs")]
            c.append(op("ss", "=" if i == 0 else "+=", "xs"))
        c += [op("ms", "=", "ss")] + self.rdiv_c("ms", S.D)
        c += [ifm("ms", rng(lo, hi), sp("inv", v)) for lo, hi, v in lut]
        for i in range(S.D):
            c += [op(dst(i), "=", src(i)), op(dst(i), "*=", "inv")] + self.rdiv_c(dst(i), S.INVR)
        return c

    # ---------------------------------------------------------------- stages
    def embed(self):
        q = self.q
        c = []
        inputs = [S.BOS] + list(range(4, S.V))              # tokens that can ever be fed in
        if self.staged:                                      # ee = E[cur], pp = P[pos], xe = ee + pp
            # dialogue: the user's line ends with <EOS>, which is FED (turn separator)
            inputs = [S.BOS, S.EOS] + list(range(4, S.V))
            dec = q.get("decode")
            if dec:
                # decoding state (spec.IntModel.generate), one unit owns all of it:
                # 'seen' flag per token (cleared at pos 0 -- and, turn_reset, whenever <EOS> is fed --
                # then set for cur); 'cp' = cur is punctuation
                fed = [t for t in inputs if t not in (S.BOS, S.EOS)]
                c += [gate("pos", 0)] + [cond(sp(f"sn{t}", 0)) for t in fed]
                if dec.get("turn_reset"):
                    c += [gate("cur", S.EOS)] + [cond(sp(f"sn{t}", 0)) for t in fed]
                c += [ifm("cur", t, sp(f"sn{t}", 1)) for t in fed]
                # cp: the token just fed is exempt (punctuation, or <EOS> = a new turn starts): spec's after_p
                c += [sp("cp", 0)] + [ifm("cur", t, sp("cp", 1)) for t in sorted(dec["exempt"])]
                c.append(SPLIT)
            if self.pack:
                c += self.packed_table("ee", "cur", inputs, q["E"], lambda i: f"ee{i}")
                c += self.packed_table("pp", "pos", range(S.T), q["P"], lambda i: f"pp{i}")
            else:
                for t in inputs:
                    c += [gate("cur", t)] + [cond(sp(f"ee{i}", q["E"][t][i])) for i in range(S.D)] + [SPLIT]
                for p in range(S.T):
                    c += [gate("pos", p)] + [cond(sp(f"pp{i}", q["P"][p][i])) for i in range(S.D)] + [SPLIT]
            for i in range(S.D):
                c += [op(f"xe{i}", "=", f"ee{i}"), op(f"xe{i}", "+=", f"pp{i}")]
            return c + [SPLIT]
        for t in inputs:
            c.append(gate("cur", t))
            c += [cond(sp(f"xe{i}", q["E"][t][i])) for i in range(S.D)] + [SPLIT]
        for p in range(S.T):
            nz = [(i, v) for i, v in enumerate(q["P"][p]) if v]
            if not nz:
                continue
            c.append(gate("pos", p))
            c += [cond(add(f"xe{i}", v)) for i, v in nz]      # xe += : must follow the embedding in one chain
        return c

    def layer(self, l):
        q = self.q
        xin = (lambda i: f"xe{i}") if l == 0 else (lambda i: nL("xf", l - 1, i))
        v = lambda nm: (lambda i: nL(nm, l, i))
        na, nb = v("na"), v("nb")
        sec = {}
        sec["RMS_A"] = self.rms(xin, na, 2 * l)
        # Q, K, V share their input: one stacked matrix (more shared pairs for CSE)
        c, rowf = self.mat(f"a{l}", q[f"Wq{l}"] + q[f"Wk{l}"] + q[f"Wv{l}"], na)
        for b, nm in enumerate(("q", "k", "v")):
            for j in range(S.D):
                c += rowf(b * S.D + j, nL(nm, l, j)) + [SPLIT]
        for j in range(S.D):
            c += [op(nL("qs", l, j), "=", nL("q", l, j))] + self.rdiv_c(nL("qs", l, j), S.QS)
            c += [op(nL("ks", l, j), "=", nL("k", l, j))] + self.rdiv_c(nL("ks", l, j), S.QS) + [SPLIT]
        sec["QKV"] = c
        c = []
        for p in range(S.T):                                    # KV cache write: one slot runs
            c.append(gate("pos", p))
            c += [cond(op(nK(l, p, i), "=", nL("ks", l, i))) for i in range(S.D)]
            c += [cond(op(nV(l, p, i), "=", nL("v", l, i))) for i in range(S.D)] + [SPLIT]
        sec["KV_WRITE"] = c
        c = []
        if self.staged:
            # causal skip: score j only when pos >= j (gate + conditional segment); masked slots are
            # set to MASKED first, exactly what the unstaged program ends up with
            for h in range(S.H):
                for j in range(S.T):
                    s, b = nS(l, h, j), S.DH * h
                    body = [op(s, "=", nL("qs", l, b)), op(s, "*=", nK(l, j, b))]
                    for i in range(b + 1, b + S.DH):
                        body += [op("tm", "=", nL("qs", l, i)), op("tm", "*=", nK(l, j, i)), op(s, "+=", "tm")]
                    body += self.rdiv_c(s, S.cst(q, "ATTDIV", l))
                    if j == 0:
                        c += body + [SPLIT]
                    else:
                        c += [sp(s, S.MASKED), gate("pos", f"{j}..")] + [cond(x) for x in body] + [SPLIT]
        else:
            for h in range(S.H):
                for j in range(S.T):
                    s, b = nS(l, h, j), S.DH * h
                    c += [op(s, "=", nL("qs", l, b)), op(s, "*=", nK(l, j, b))]
                    for i in range(b + 1, b + S.DH):
                        c += [op("tm", "=", nL("qs", l, i)), op("tm", "*=", nK(l, j, i)), op(s, "+=", "tm")]
                    c += self.rdiv_c(s, S.cst(q, "ATTDIV", l)) + [SPLIT]
                for j in range(1, S.T):
                    c.append(ifm("pos", rng(None, j - 1), sp(nS(l, h, j), S.MASKED)))
        sec["ATTN_SCORE"] = c
        c = []
        for h in range(S.H):
            c.append(op("smx", "=", nS(l, h, 0)))
            c += [op("smx", ">", nS(l, h, j)) for j in range(1, S.T)]
            for j in range(S.T):
                w = nW(l, h, j)
                c += [op("dl", "=", nS(l, h, j)), op("dl", "-=", "smx"), sp(w, 0)]
                c += [ifm("dl", rng(lo, hi), sp(w, val)) for lo, hi, val in q["exp_lut"]]
            den, off = nDen(l, h), nOff(l, h)
            c.append(op(den, "=", nW(l, h, 0)))
            c += [op(den, "+=", nW(l, h, j)) for j in range(1, S.T)]
            c += [op(off, "=", den), op(off, "*=", self.const(S.KATT), CO),
                  op("hd", "=", den), op("hd", "/=", self.const(2), CO), op(off, "+=", "hd"), SPLIT]
        sec["ATTN_SOFTMAX"] = c
        c = []
        for h in range(S.H if self.staged else 0):
            # causal skip: one unit per head; slot j contributes only when pos >= j (w_j is 0 otherwise)
            den, off = nDen(l, h), nOff(l, h)
            hi = range(S.DH * h, S.DH * h + S.DH)
            for i in hi:
                c += [op(nL("a", l, i), "=", nV(l, 0, i)), op(nL("a", l, i), "*=", nW(l, h, 0))]
            for j in range(1, S.T):
                c.append(gate("pos", f"{j}.."))
                for i in hi:
                    c += [cond(op("tm", "=", nV(l, j, i))), cond(op("tm", "*=", nW(l, h, j))), cond(op(nL("a", l, i), "+=", "tm"))]
            for i in hi:
                a = nL("a", l, i)
                c += [op(a, "+=", off), op(a, "/=", den), add(a, -S.KATT)]
            c.append(SPLIT)
        for h in range(S.H if not self.staged else 0):
            den, off = nDen(l, h), nOff(l, h)
            for i in range(S.DH * h, S.DH * h + S.DH):
                a = nL("a", l, i)
                c += [op(a, "=", nV(l, 0, i)), op(a, "*=", nW(l, h, 0))]
                for j in range(1, S.T):
                    c += [op("tm", "=", nV(l, j, i)), op("tm", "*=", nW(l, h, j)), op(a, "+=", "tm")]
                c += [op(a, "+=", off), op(a, "/=", den), add(a, -S.KATT), SPLIT]
        sec["ATTN_VALUE"] = c
        c, rowf = self.mat(f"o{l}", q[f"Wo{l}"], v("a"))
        for j in range(S.D):
            c += rowf(j, "o")
            c += [op("o", "*=", self.const(S.cst(q, "MO", l)), CO)] + self.rdiv_c("o", q.get("R", S.R))
            c += [op(nL("xa", l, j), "=", xin(j)), op(nL("xa", l, j), "+=", "o"), SPLIT]
        sec["O_PROJ_RESIDUAL"] = c
        sec["RMS_B"] = self.rms(v("xa"), nb, 2 * l + 1)
        c, rowf = self.mat(f"f{l}", q[f"W1{l}"], nb)
        for j in range(S.F):
            c += rowf(j, nL("f", l, j))
            if self.staged:                                # ReLU inside the unit that writes f
                c.append(ifm(nL("f", l, j), "..-1", sp(nL("f", l, j), 0)))
            c.append(SPLIT)
        if not self.staged:
            c += [ifm(nL("f", l, j), "..-1", sp(nL("f", l, j), 0)) for j in range(S.F)]
        sec["FF_UP_RELU"] = c
        c, rowf = self.mat(f"g{l}", q[f"W2{l}"], v("f"))
        for j in range(S.D):
            c += rowf(j, "g")
            c += [op("g", "*=", self.const(S.cst(q, "M2", l)), CO)] + self.rdiv_c("g", q.get("R", S.R))
            c += [op(nL("xf", l, j), "=", nL("xa", l, j)), op(nL("xf", l, j), "+=", "g"), SPLIT]
        sec["FF_DOWN_RESIDUAL"] = c
        return sec

    def head(self):
        q = self.q
        nf = lambda i: f"nf{i}"
        c = self.rms(lambda i: nL("xf", S.L - 1, i), nf, 2 * S.L)
        if self.staged:
            # block-local argmax (hb/hv per block of HB tokens, SSA), then merge in token order:
            # strict '>' keeps the smaller token id on ties, like the single running argmax
            c.append(SPLIT)
            al = q["allowed"]
            blocks = [al[k:k + self.HB] for k in range(0, len(al), self.HB)]
            dec = q.get("decode")
            ex = set(dec["exempt"]) if dec else set()
            prep, rowf = self.mat("h", [q["Wh"][t] for t in al], nf)
            c += prep
            ai = {t: k for k, t in enumerate(al)}
            for b, toks in enumerate(blocks):
                hb, hv = f"hb{b}", f"hv{b}"
                for idx, t in enumerate(toks):
                    c += rowf(ai[t], "lg")
                    if dec and t not in ex:                  # presence penalty for already-seen tokens
                        c.append(ifm(f"sn{t}", 1, add("lg", -dec["penalty"])))
                    if dec and t in ex and t != S.EOS:       # no punctuation right after punctuation
                        c.append(ifm("cp", 1, add("lg", -S.BAN)))
                    if idx == 0:
                        c += [op(hv, "=", "lg"), sp(hb, t)]
                    else:
                        c += [f"execute if score lg {OBJ} > {hv} {OBJ} run " + sp(hb, t), op(hv, ">", "lg")]
                c.append(SPLIT)
            c += [op("bt", "=", "hb0"), op("blg", "=", "hv0")]
            for b in range(1, len(blocks)):
                c += [f"execute if score hv{b} {OBJ} > blg {OBJ} run " + op("bt", "=", f"hb{b}"), op("blg", ">", f"hv{b}")]
            return c + [SPLIT]
        for idx, t in enumerate(q["allowed"]):
            c += self.tern_row("lg", q["Wh"][t], nf)
            if idx == 0:
                c += [op("blg", "=", "lg"), sp("bt", t)]
            else:
                c += [f"execute if score lg {OBJ} > blg {OBJ} run " + sp("bt", t), op("blg", ">", "lg")]
            c.append(SPLIT)
        return c

    def control(self):
        """after the forward at (cur, pos): feed back the next token, print generated ones, stop"""
        q = self.q
        c = [op("kk", "=", "pos"), add("kk", 1), sp("gen", 0),
             f"execute if score kk {OBJ} >= plen {OBJ} run " + sp("gen", 1),
             op("nxt", "=", "bt")]
        c += [f"execute if score gen {OBJ} matches 0 if score pos {OBJ} matches {j} run " + op("nxt", "=", f"t{j+1}")
              for j in range(S.T - 1)]
        c += [f"execute if score gen {OBJ} matches 1 if score pos {OBJ} matches {p} run " + op(f"out{p}", "=", "bt")
              for p in range(S.T)]
        for t in q["allowed"]:
            txt = "§7<EOS>" if t == S.EOS else "§e" + self.vocab[t]
            c.append(f"execute if score gen {OBJ} matches 1 if score bt {OBJ} matches {t} run " + tell(f"§b[{S.label()}]§r " + txt))
        c += [f"execute if score gen {OBJ} matches 1 if score bt {OBJ} matches {S.EOS} run " + sp("run", 0),
              ifm("kk", f"{S.T}..", sp("run", 0)),
              op("cur", "=", "nxt"), op("pos", "=", "kk"), add("fw", 1)]
        # snapshot of the final forward for the acceptance check (later stray forwards may overwrite xf/bt)
        c += [ifm("run", 0, op(snap(n), "=", n)) for n in snap_names()]
        c.append(ifm("run", 0, tell(f"§a[{S.label()}] 生成结束")))
        return [guard(*self.ctl_guard, x) for x in c]

    # ---------------------------------------------------------------- programs
    def groups(self, single=False):
        """[(group name, [(section name, [cmd])])]
        single=False: three tick groups, control guarded by grp == 2 (controller-driven layout)
        single=True : one chain = one forward per tick; 'live = run' latched first, control guarded
                      by live == 1 (a repeating head keeps running 1-2 ticks after its power goes)"""
        self.ctl_guard = ("live", 1) if single else ("grp", 2)
        g0 = ([("LATCH", [op("live", "=", "run")])] if single else []) + [("EMBED", self.embed())] + list(self.layer(0).items())
        g1 = list(self.layer(1).items())
        g2 = [("HEAD_ARGMAX", self.head()), ("CONTROL", self.control())]
        return [("G0", g0), ("G1", g1), ("G2", g2)]

    def stages(self, budget=60000):
        """staged forward (staged=True): [(stage name, [(section, cmds)])], one chain per stage and
        <= budget commands each before regate. Stage k's tail (layout.build_plan_staged) powers
        stage k+1 only while stg == k, so a stray second firing of a chain cannot start a second
        pipeline; CONTROL (the only non-SSA part) runs in the last stage, guarded by stg == last."""
        assert self.staged, "stages() needs Compiler(staged=True)"
        secs = [("EMBED", self.embed())]
        for l in range(S.L):
            secs += [(f"L{l}_{n}", c) for n, c in self.layer(l).items()]
        secs.append(("HEAD_ARGMAX", self.head()))
        units = []
        for name, cmds in secs:
            cur = []
            for c in cmds + [SPLIT]:
                if c == SPLIT:
                    if cur:
                        units.append((name, cur))
                    cur = []
                else:
                    cur.append(c)
        out, cur, n = [], [], 0

        def put(name, u):
            nonlocal cur, n
            if n + len(u) > budget and cur:
                out.append(cur); cur, n = [], 0
            if cur and cur[-1][0] == name:
                cur[-1][1].extend(u)
            else:
                cur.append((name, list(u)))
            n += len(u)
        for name, u in units:
            put(name, u)
        self.ctl_guard = ("stg", 0)
        nctl = len(self.control())
        last = len(out) + (1 if n + nctl > budget and cur else 0)       # the stage CONTROL lands in
        self.ctl_guard = ("stg", last)
        put("CONTROL", self.control())
        out.append(cur)
        assert len(out) == last + 1
        return [(f"G{k}", s) for k, s in enumerate(out)]

    def single_staged(self):
        """the staged program as ONE chain (one forward per tick; needs maxcommandchainlength raised,
        measured in-game: 100,000 in one chain runs completely with the gamerule at 1,000,000).
        Same self-contained units; control guarded by the live latch like T44K single mode."""
        assert self.staged
        secs = [("LATCH", [op("live", "=", "run")]), ("EMBED", self.embed())]
        for l in range(S.L):
            secs += [(f"L{l}_{n}", c) for n, c in self.layer(l).items()]
        secs.append(("HEAD_ARGMAX", self.head()))
        self.ctl_guard = ("live", 1)
        secs.append(("CONTROL", self.control()))
        return [("G", secs)]

    def init_program(self):
        c = [f"scoreboard objectives add {OBJ} dummy", f"scoreboard objectives add {CO} dummy"]
        c += [f"scoreboard players set {k} {CO} {v}" for k, v in sorted(self.consts.items())]
        c += [sp(n, 0) for n in ("run", "grp", "arm", "go", "tk", "pos", "fw", "plen", "live")] + [sp("cur", S.BOS)]
        if not self.staged:
            # staged builds read KV slot j only when pos >= j, after the slot was written at pos j:
            # zeroing the cache is not needed (T4M: 196K init commands = 196K command blocks saved;
            # the strict VM of check_staged faults on any read of an unassigned score)
            c += [sp(f(l, p, i), 0) for f in (nK, nV) for l in range(S.L) for p in range(S.T) for i in range(S.D)]
        c += [sp(f"t{j}", 0) for j in range(S.T)] + [sp(f"out{p}", -1) for p in range(S.T)]
        if self.staged:
            c.append(sp("stg", 0))
        return c

    def start_program(self, prompt_ids):
        """set prompt, reset position, start (grp 0, armed)"""
        c = [sp("plen", len(prompt_ids))] + [sp(f"t{j}", t) for j, t in enumerate(prompt_ids)]
        c += [sp(f"out{p}", -1) for p in range(S.T)]
        c += [sp("pos", 0), op("cur", "=", "t0"), sp("grp", 0), sp("arm", 1), sp("fw", 0), sp("run", 1)]
        if self.staged:
            c.append(sp("stg", 0))
        return c


def flatten(sections):
    return [c for _, s in sections for c in s if c != SPLIT]


def stats(groups):
    out = {}
    for g, secs in groups:
        cmds = flatten(secs)
        out[g] = {"commands": len(cmds), "conditional": sum(c.startswith("?") for c in cmds),
                  "by_section": {n: len(s) for n, s in secs}}
    return out
