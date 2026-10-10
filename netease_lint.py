"""NetEase's sensitive-word filter silently drops any command whose text contains a blocked
word: no error, no feedback, the command just does nothing (found the hard way: E1 used the
scoreboard objective 'sb', and every command touching it vanished).

check() runs over every command a build emits and fails the build on a risky substring.
The list is conservative (pinyin insult abbreviations and the usual political numbers); add to
it whenever the game swallows something.
"""
import re

RISKY = ["sb", "cnm", "nmsl", "nmb", "tmd", "mlgb", "wcnm", "rnm", "sima", "gcd", "8964", "六四", "法轮"]
_pat = re.compile("|".join(re.escape(w) for w in RISKY), re.IGNORECASE)


# Score-holder NAMES are filtered more aggressively (probably treated like player names), and
# the filter seems to ignore separators. Measured in-game (2026-09-29): dropped
#   K0_7_31, K0_9_18, K1_7_31, K1_9_18, a6_4_89, a89p64, a8964, bl, s0h0j8 / w1h3j8 (any '...j8')
# while y731, y918, K0p7d31, K0p9d18, a64, a89, a1989 pass. The in-game probe of every name
# (t44k ingame builder) is the authority; this list only catches known cases early.
# "qq": every score holder 'qqqqqqqqqqqqq<n>' was rejected in-game (2026-10-02 micro-benchmark)
NAME_RISKY = [r"j8", r"^bl$", r"8964", r"6489", r"_7_31", r"_9_18", r"89\D*64", r"qq"]
_npat = re.compile("|".join(NAME_RISKY), re.IGNORECASE)


def check_names(names, where=""):
    bad = [n for n in names if _npat.search(n) or _pat.search(n)]
    if bad:
        raise ValueError(f"{where}: {len(bad)} score-holder names look filtered: {bad[:10]}")


def check(commands, where=""):
    bad = [(c, _pat.search(c).group(0)) for c in commands if _pat.search(c)]
    if bad:
        sample = "\n  ".join(f"[{w}] {c[:120]}" for c, w in bad[:5])
        raise ValueError(f"{where}: {len(bad)} commands contain NetEase-filtered words:\n  {sample}")
