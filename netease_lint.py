"""NetEase's sensitive-word filter silently drops any command whose text contains a blocked
word: no error, no feedback, the command just does nothing (found the hard way: E1 used the
scoreboard objective 'sb', and every command touching it vanished).

check() runs over every command a build emits and fails the build on a risky substring.
The list is conservative (pinyin insult abbreviations and words / digit strings the platform
blocks); add to it whenever the game swallows something. The lists are stored base64-encoded
so that this source file itself carries no filtered words (print RISKY / NAME_RISKY to see them).
"""
import re, json, base64


def _dec(s):
    return json.loads(base64.b64decode(s).decode("utf-8"))


RISKY = _dec("WyJzYiIsICJjbm0iLCAibm1zbCIsICJubWIiLCAidG1kIiwgIm1sZ2IiLCAid2NubSIsICJybm0iLCAic2ltYSIsICJnY2QiLCAiODk2NCIsICLlha3lm5siLCAi5rOV6L2uIl0=")
_pat = re.compile("|".join(re.escape(w) for w in RISKY), re.IGNORECASE)


# Score-holder NAMES are filtered more aggressively (probably treated like player names), and
# the filter seems to ignore separators: names whose digits form certain dates or numbers were
# dropped even when split by '_' or a letter, as were 'bl', any '...j8' and 'qq...' (in-game
# probes 2026-09-29 / 10-02). The in-game probe of every name (t44k ingame builder) is the
# authority; this list only catches known cases early.
NAME_RISKY = _dec("WyJqOCIsICJeYmwkIiwgIjg5NjQiLCAiNjQ4OSIsICJfN18zMSIsICJfOV8xOCIsICI4OVxcRCo2NCIsICJxcSJd")
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
