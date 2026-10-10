"""Run a module under a hard memory cap (t44k/memcap.py): a job that needs more fails with MemoryError
instead of freezing the computer.

  python -m t44k.capped GB module [args...]        e.g.  python -m t44k.capped 11 t44k.check_staged t10m 2 ...
  python -m t44k.capped GB script.py [args...]
"""
import sys, os, runpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from t44k.memcap import cap_memory

if __name__ == "__main__":
    cap_memory(float(sys.argv[1]))
    mod = sys.argv[2]
    sys.argv = [mod] + sys.argv[3:]
    if mod.endswith(".py"):
        runpy.run_path(mod, run_name="__main__")
    else:
        runpy.run_module(mod, run_name="__main__", alter_sys=True)
