"""Offline test of the M7 pinyin input: run the IME chain's command TEXT (vm.VM) for scripted key
presses and check the trie state, pages, picks, prompt buffer, backspace / clear / generate.

  python -m t44k.sim_m7 EXPORT_DIR
"""
import sys, json
from vm import VM
from t44k import spec as S
from t44k.compiler import OBJ
from t44k.m7 import build_m7, PAGE


class M7VM(VM):
    def run(self, cmd):
        if cmd.startswith("titleraw "):
            self.executed += 1; self.effective += 1
            self.bar = "".join(p.get("text", "") for p in json.loads(cmd.split(" ", 3)[3])["rawtext"])
            return True
        if cmd.startswith("clone "):                   # chain breaker restore (layout repeat_brk)
            self.executed += 1; self.effective += 1
            self.blocks[tuple(cmd.split(" ")[7:10])] = "restored"
            return True
        return super().run(cmd)


def main(export_dir, dialog=False):
    m = build_m7(export_dir, dialog)
    ime, init = m["chains"]["ime"], m["chains"]["init"]
    node, cands, syls = m["tables"]
    keys = {k["label"]: k["cmds"] for k in m["keys"]}
    vm = M7VM()
    vm.bar = ""
    vm.run_all(init)
    g = lambda n: vm.s.get((n, OBJ))
    fails = []

    def press(label):
        vm.out.clear()
        vm.run_all(keys[label])
        vm.run_all(m["chains"]["watchdog"])            # the always-active watchdog sees the pending key
        assert vm.blocks.get(tuple(str(x) for x in m["ime_power"])) == "redstone_block"
        vm.run_all(ime)
        assert vm.blocks.get(tuple(str(x) for x in m["ime_power"])) == "air"

    def check(what, cond):
        if not cond:
            fails.append(what)
        print(("ok   " if cond else "FAIL ") + what)

    def typ(s):
        for L in s:
            press(L.upper())

    typ("shi")
    check(f"s-h-i -> node shi, bar: {vm.bar}", g("imn") == node["shi"] and vm.bar.startswith("shi  [1/"))
    n_pages = (len(cands[node["shi"]]) + PAGE - 1) // PAGE
    press("下页"); check("下页 -> page 2", g("imp") == 1 and "[2/" in vm.bar)
    for _ in range(n_pages + 2):
        press("下页")
    check(f"下页 clamps at the last page ({n_pages})", g("imp") == n_pages - 1)
    for _ in range(n_pages + 2):
        press("上页")
    check("上页 clamps at page 1", g("imp") == 0)
    want1 = cands[node["shi"]][0]
    press("1")
    check(f"pick 1 -> t1 = {want1}, plen 2, pinyin cleared; echo {vm.out}", g("t1") == want1 and g("plen") == 2 and g("imn") == 0)
    typ("x"); press("退格")
    check("x + 退格 -> pinyin empty, prompt kept", g("imn") == 0 and g("plen") == 2)
    press("退格")
    check("退格 on empty pinyin deletes the last character", g("plen") == 1)
    typ("qq")
    check(f"q-q: invalid second q is ignored with a message {vm.out}", g("imn") == node["q"] and any("没有" in o for o in vm.out))
    press("清空")
    typ("zhong"); press("1"); typ("guo"); press("2")
    want = [cands[node["zhong"]][0], cands[node["guo"]][1]]
    check(f"zhong 1, guo 2 -> t1 t2 = {want}", [g("t1"), g("t2")] == want and g("plen") == 3)
    # every character is reachable through its full syllable
    vocab = json.load(open("data/mixd/vocab.json" if dialog else "data/mix/vocab.json", encoding="utf-8"))["vocab"]
    reach = {t for n, lst in cands.items() for t in lst}
    missing = [vocab[i] for i in range(4, S.V) if "一" <= vocab[i] <= "鿿" and i not in reach]
    check(f"all characters reachable ({len(missing)} missing)", not missing)
    press("1")
    check("digit without pinyin -> hint, nothing appended", g("plen") == 3 and any("先按字母" in o for o in vm.out))
    if dialog:
        return dialog_turns(m, vm, g, press, typ, check, node, cands, fails, ime)
    press("生成")
    check(f"生成 -> run 1, pos 0, cur = t0, core powered", g("run") == 1 and g("pos") == 0 and g("cur") == S.BOS
          and vm.blocks.get(tuple(str(x) for x in m["core_power"])) == "redstone_block" and g("out0") == -1)
    typ("de"); press("1")
    check("pick while generating -> refused", g("plen") == 3 and any("正在生成" in o for o in vm.out))
    vm.s[("run", OBJ)] = 0
    press("自检")
    tp = m["selftest"]["prompt"]
    check("自检 -> fixed prompt, st 1, run 1", [g(f"t{j}") for j in range(len(tp))] == tp and g("st") == 1 and g("run") == 1)
    print("M7 IME OFFLINE CHECK", "PASS" if not fails else f"FAIL {fails}", f"(IME chain {len(ime)} commands)")
    return not fails


def dialog_turns(m, vm, g, press, typ, check, node, cands, fails, ime):
    """two turns: send starts at pos 0; after the reply (simulated: run 0, pos 7, cur EOS) the next
    characters go after the reply's EOS position and send CONTINUES from pos 7"""
    P = tuple(str(x) for x in m["core_power"])
    # the prompt is t1 t2 (zhong guo), plen 3
    press("发送")
    check("发送 #1 -> <EOS> appended, plen 4, fresh start (pos 0, cur BOS), cv 1, nt 1",
          g("t3") == S.EOS and g("plen") == 4 and g("run") == 1 and g("pos") == 0 and g("cur") == S.BOS
          and g("cv") == 1 and g("nt") == 1 and vm.blocks.get(P) == "redstone_block")
    press("发送")
    check("发送 while answering -> refused", g("plen") == 4 and any("正在回答" in o for o in vm.out))
    vm.s[("run", OBJ)] = 0; vm.s[("pos", OBJ)] = 7; vm.s[("cur", OBJ)] = S.EOS      # the reply ended at pos 7
    vm.s[("out3", OBJ)] = 55; vm.blocks[P] = "air"
    press("发送")
    check("发送 with nothing typed in the new turn -> refused", g("plen") == 4 and g("run") == 0)
    typ("ni"); press("1")
    w = cands[node["ni"]][0]
    check(f"first character of turn 2 -> t8 = {w}, plen 9, ts 8, nt 0", g("t8") == w and g("plen") == 9
          and g("ts") == 8 and g("nt") == 0)
    press("退格")
    check("退格 inside the turn deletes it (plen 8)", g("plen") == 8)
    press("退格")
    check("退格 at the turn start keeps the history (plen 8)", g("plen") == 8)
    typ("hao"); press("1")
    press("发送")
    check("发送 #2 -> t9 = <EOS>, plen 10, CONTINUE (pos 7, cur EOS kept), out reset, run 1",
          g("t9") == S.EOS and g("plen") == 10 and g("run") == 1 and g("pos") == 7 and g("cur") == S.EOS
          and g("out3") == -1 and vm.blocks.get(P) == "redstone_block")
    vm.s[("run", OBJ)] = 0
    press("清空")
    check("清空 -> new conversation (plen 1, cv 0, nt 0, ts 1)", g("plen") == 1 and g("cv") == 0 and g("nt") == 0 and g("ts") == 1)
    print("M7 IME OFFLINE CHECK (dialogue)", "PASS" if not fails else f"FAIL {fails}", f"(IME chain {len(ime)} commands)")
    return not fails


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], "--dialog" in sys.argv) else 1)
