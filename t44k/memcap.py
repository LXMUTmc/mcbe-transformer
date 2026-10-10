"""Hard per-process memory cap on Windows (Job Object), so a big offline job fails with MemoryError
instead of freezing the user's PC (2026-10-08: an uncapped job did)."""
import ctypes, sys
import ctypes.wintypes as wt


def cap_memory(gb):
    if sys.platform != "win32":                     # Linux / macOS: address-space limit
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (int(gb * 1024 ** 3), int(gb * 1024 ** 3)))
        return None

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rb", "wb", "ob")]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD), ("SchedulingClass", wt.DWORD)]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("Basic", BASIC), ("Io", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wt.HANDLE
    k32.GetCurrentProcess.restype = wt.HANDLE
    job = k32.CreateJobObjectW(None, None)
    info = EXTENDED()
    info.Basic.LimitFlags = 0x100 | 0x2000          # PROCESS_MEMORY | KILL_ON_JOB_CLOSE
    info.ProcessMemoryLimit = int(gb * 1024 ** 3)
    ok = k32.SetInformationJobObject(wt.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info))
    ok = ok and k32.AssignProcessToJobObject(wt.HANDLE(job), wt.HANDLE(k32.GetCurrentProcess()))
    if not ok:
        sys.exit(f"could not cap memory (winerr {ctypes.get_last_error()}); refusing to run uncapped")
    return job
