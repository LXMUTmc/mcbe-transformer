# -*- coding: utf-8 -*-
"""MCBE Transformer 工坊: train a model on any mix of corpora and turn it into a Minecraft world.

A thin window (tkinter, standard library) over t44k/project.py: every page only edits
projects/NAME/project.json; the work runs in a separate process (python -m t44k.project run ...)
whose progress the window reads from projects/NAME/status.json and train/progress.json.

  .venv\\Scripts\\pythonw.exe app\\mct_studio.py        (or 启动工坊.bat)
"""
import os, sys, json, subprocess, threading, webbrowser, shutil, time, ctypes
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PROJECTS = os.path.join(ROOT, "projects")
PY = sys.executable.replace("pythonw.exe", "python.exe")
NOWIN = 0x08000000 if sys.platform == "win32" else 0          # CREATE_NO_WINDOW

# model sizes: params, game command blocks, seconds per character, game RAM (GB), training seconds per
# step (RTX 4070 SUPER, torch.compile), default steps -- measured (mini: estimated)
SIZES = [
    ("mini",  "迷你",  217_088,   60_000,   0.1,  1.5, 0.010,   3000),
    ("t1m64", "T1M",   1_253_376, 850_000,  3.0,  4.5, 0.034,  60000),
    ("t4m",   "T4M",   4_472_832, 1_800_000, 5.3, 7.0, 0.094,  80000),     # 80K steps: 2 h 5 min measured (2026-10-10)
    ("t7m",   "T7M",   6_819_840, 2_440_000, 7.9, 8.0, 0.130, 100000),
    ("t10m",  "T10M",  9_658_368, 3_107_262, 11.0, 9.3, 0.169, 160000),
]
SOURCES = {
    "teacher": ("DeepSeek 老师语料", "书面语短句，质量最好；需要你自己的 DeepSeek 密钥，约 10 美元 / 215 万句"),
    "wiki":    ("中文维基百科", "百科知识句子；CC BY-SA 3.0；约 500 MB"),
    "lccc":    ("LCCC 对话", "真人短对话（微博等），让模型会聊天；作者声明仅限研究用途；约 350 MB"),
    "text":    ("我自己的文本", "小说、文章、聊天记录……放 txt 进来；对话用 | 分隔轮次，或一行一句、空行分段"),
}
DEFAULT_W = {"teacher": 0.40, "wiki": 0.15, "lccc": 0.45, "text": 0.30}
LINKS = {"wiki": "https://huggingface.co/datasets/0xDing/wikipedia-cn-20230720-filtered",
         "lccc": "https://huggingface.co/datasets/silver/lccc",
         "deepseek": "https://platform.deepseek.com/api_keys"}


def human(s):
    s = int(s)
    return f"{s // 3600} 小时 {s % 3600 // 60} 分" if s >= 3600 else f"{s // 60} 分 {s % 60} 秒"


def mem_gb():
    class M(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                    ("pt", ctypes.c_ulonglong), ("pa", ctypes.c_ulonglong), ("vt", ctypes.c_ulonglong), ("va", ctypes.c_ulonglong),
                    ("x", ctypes.c_ulonglong)]
    m = M(); m.l = ctypes.sizeof(M)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.total / 2**30, m.avail / 2**30


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MCBE Transformer 工坊")
        self.geometry("980x700")
        self.proc = None
        self.env = {}
        style = ttk.Style(self)
        style.configure("H.TLabel", font=("Microsoft YaHei UI", 13, "bold"))
        style.configure("S.TLabel", foreground="#666")
        top = ttk.Frame(self, padding=8); top.pack(fill="x")
        ttk.Label(top, text="项目：").pack(side="left")
        self.proj = tk.StringVar()
        self.proj_box = ttk.Combobox(top, textvariable=self.proj, width=24, state="readonly")
        self.proj_box.pack(side="left")
        self.proj_box.bind("<<ComboboxSelected>>", lambda e: self.load_project())
        ttk.Button(top, text="新建项目", command=self.new_project).pack(side="left", padx=6)
        ttk.Button(top, text="保存", command=self.save_project).pack(side="left")
        ttk.Label(top, text="  每个项目一个文件夹：projects\\项目名", style="S.TLabel").pack(side="left")
        nb = ttk.Notebook(self); nb.pack(fill="both", expand=True, padx=8, pady=4)
        self.pages = {}
        for key, title in (("pc", "① 电脑"), ("data", "② 语料"), ("model", "③ 模型"), ("run", "④ 训练与生成"), ("chat", "⑤ 试聊")):
            f = ttk.Frame(nb, padding=12); nb.add(f, text=title); self.pages[key] = f
        self.build_pc(); self.build_data(); self.build_model(); self.build_run(); self.build_chat()
        self.refresh_projects()
        self.after(300, self.check_env_async)
        self.after(1000, self.poll)

    # ---------------------------------------------------------------- ① PC
    def build_pc(self):
        f = self.pages["pc"]
        ttk.Label(f, text="电脑检查", style="H.TLabel").pack(anchor="w")
        self.pc_text = tk.Text(f, height=14, width=110, font=("Consolas", 10)); self.pc_text.pack(fill="x", pady=6)
        b = ttk.Frame(f); b.pack(anchor="w")
        ttk.Button(b, text="重新检查", command=self.check_env_async).pack(side="left")
        ttk.Button(b, text="安装 / 修复依赖（PyTorch 等，需联网，约 3 GB）", command=self.install_deps).pack(side="left", padx=8)
        ttk.Label(f, text="训练要 NVIDIA 显卡（8 GB 显存以上较好）；游戏里跑模型主要吃内存：T10M 需要游戏进程约 9 GB，建议 32 GB 内存。",
                  style="S.TLabel", wraplength=900).pack(anchor="w", pady=8)

    def check_env_async(self):
        self.pc_text.delete("1.0", "end"); self.pc_text.insert("end", "检查中……\n")
        threading.Thread(target=self.check_env, daemon=True).start()

    def check_env(self):
        code = ("import json,sys,shutil\nr={'python':sys.version.split()[0]}\n"
                "try:\n import torch;r['torch']=torch.__version__;r['cuda']=torch.cuda.is_available()\n"
                " if r['cuda']:\n  p=torch.cuda.get_device_properties(0);r['gpu']=p.name;r['vram']=round(p.total_memory/2**30,1)\n"
                "except Exception as e: r['torch']=None\n"
                "for m in ('numpy','pypinyin','triton'):\n try:\n  __import__(m);r[m]=True\n except Exception: r[m]=False\n"
                "print(json.dumps(r))")
        try:
            out = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=120, creationflags=NOWIN).stdout
            r = json.loads(out.strip().splitlines()[-1])
        except Exception as e:
            r = {"error": str(e)}
        total, avail = mem_gb()
        disk = shutil.disk_usage(ROOT).free / 2**30
        r.update(ram=round(total, 1), ram_free=round(avail, 1), disk=round(disk, 1))
        self.env = r
        lines = [f"Python      {r.get('python', '?')}",
                 f"PyTorch     {r.get('torch') or '未安装（点下面的按钮安装）'}",
                 f"显卡        {r.get('gpu', '没有可用的 NVIDIA 显卡（只能用 CPU，训练会非常慢）')}" + (f"，{r['vram']} GB 显存" if r.get("vram") else ""),
                 f"加速编译    {'可用（训练快约 2 倍）' if r.get('triton') else '不可用（可以训练，只是慢一些）'}",
                 f"其他依赖    numpy {'✓' if r.get('numpy') else '✗'}   pypinyin {'✓' if r.get('pypinyin') else '✗'}",
                 f"内存        {r['ram']} GB（当前空闲 {r['ram_free']} GB）",
                 f"硬盘空闲    {r['disk']} GB（语料 + 训练 + 生成，T10M 全套约 10 GB）", ""]
        fit = [lbl for k, lbl, p, b, s, g, t, st in SIZES if g + 6 <= r["ram"]]
        lines.append(f"建议：这台电脑在游戏里最大能跑 {fit[-1] if fit else '迷你'}（游戏内存 + 系统约 6 GB 余量）")
        self.after(0, lambda: (self.pc_text.delete("1.0", "end"), self.pc_text.insert("end", "\n".join(lines))))

    def install_deps(self):
        if not messagebox.askyesno("安装依赖", "将用 pip 安装 numpy、pypinyin、PyTorch（CUDA 12.8 版，约 3 GB）和 triton-windows。\n需要联网，可能要十几分钟。继续吗？"):
            return
        cmds = [[PY, "-m", "pip", "install", "numpy", "pypinyin"],
                [PY, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu128"],
                [PY, "-m", "pip", "install", "triton-windows<3.7"]]
        self.run_console(cmds, "安装依赖", after=self.check_env_async)

    def run_console(self, cmds, title, after=None, env=None):
        """run commands in a visible console window (pip / downloads show their own progress)"""
        def go():
            for c in cmds:
                subprocess.run(["cmd", "/c"] + c + ["&&", "echo.", "&&", "pause"] if c is cmds[-1] else c,
                               creationflags=subprocess.CREATE_NEW_CONSOLE, env=env, cwd=ROOT)
            if after:
                self.after(0, after)
        threading.Thread(target=go, daemon=True).start()

    # ---------------------------------------------------------------- ② data
    def build_data(self):
        f = self.pages["data"]
        ttk.Label(f, text="选语料（可以任意组合，比例 = 每批训练数据里各占多少）", style="H.TLabel").pack(anchor="w")
        self.src = {}
        for k, (name, desc) in SOURCES.items():
            box = ttk.LabelFrame(f, text=name, padding=6); box.pack(fill="x", pady=4)
            on, w = tk.BooleanVar(value=k in ("wiki", "lccc")), tk.DoubleVar(value=DEFAULT_W[k])
            row = ttk.Frame(box); row.pack(fill="x")
            ttk.Checkbutton(row, text="使用", variable=on).pack(side="left")
            ttk.Label(row, text="  比例").pack(side="left")
            ttk.Spinbox(row, from_=0.05, to=1.0, increment=0.05, textvariable=w, width=6).pack(side="left")
            ttk.Label(row, text="   " + desc, style="S.TLabel").pack(side="left")
            path = tk.StringVar()
            r2 = ttk.Frame(box); r2.pack(fill="x", pady=2)
            d = {"on": on, "w": w, "path": path}
            if k == "wiki":
                path.set(os.path.join(ROOT, "data", "wiki_raw.json"))
                ttk.Entry(r2, textvariable=path, width=80).pack(side="left")
                ttk.Button(r2, text="选择文件", command=lambda p=path: self.pick_file(p, [("JSON", "*.json")])).pack(side="left", padx=4)
                ttk.Button(r2, text="打开下载页", command=lambda: webbrowser.open(LINKS["wiki"])).pack(side="left")
                ttk.Label(box, text="下载 wikipedia-cn-20230720-filtered.json，放到上面的路径（或改路径）", style="S.TLabel").pack(anchor="w")
            elif k == "lccc":
                path.set(os.path.join(ROOT, "data", "lccc"))
                ttk.Entry(r2, textvariable=path, width=80).pack(side="left")
                ttk.Button(r2, text="选择文件夹", command=lambda p=path: self.pick_dir(p)).pack(side="left", padx=4)
                ttk.Button(r2, text="打开下载页", command=lambda: webbrowser.open(LINKS["lccc"])).pack(side="left")
                d["agree"] = tk.BooleanVar(value=False)
                d["dedull"] = tk.BooleanVar(value=True)
                r3 = ttk.Frame(box); r3.pack(fill="x")
                ttk.Checkbutton(r3, text="我知道 LCCC 只能用于研究和非商业用途", variable=d["agree"]).pack(side="left")
                ttk.Checkbutton(r3, text="去掉万能回答（哈哈 / 我也是……，推荐）", variable=d["dedull"]).pack(side="left", padx=12)
                ttk.Label(box, text="文件夹里要有 lccc_base_train.jsonl.gz、lccc_base_valid.jsonl.gz、lccc_base_test.jsonl.gz", style="S.TLabel").pack(anchor="w")
            elif k == "teacher":
                path.set(os.path.join(ROOT, "data", "teacher2"))
                ttk.Entry(r2, textvariable=path, width=80).pack(side="left")
                ttk.Button(r2, text="选择文件夹", command=lambda p=path: self.pick_dir(p)).pack(side="left", padx=4)
                r3 = ttk.Frame(box); r3.pack(fill="x", pady=2)
                d["key"], d["n"] = tk.StringVar(), tk.IntVar(value=200000)
                ttk.Label(r3, text="还没有？DeepSeek 密钥").pack(side="left")
                ttk.Entry(r3, textvariable=d["key"], show="*", width=28).pack(side="left")
                ttk.Label(r3, text=" 句数").pack(side="left")
                ttk.Spinbox(r3, from_=10000, to=3000000, increment=50000, textvariable=d["n"], width=9).pack(side="left")
                ttk.Button(r3, text="开始生成（可随时关窗口，下次接着生成）", command=self.gen_teacher).pack(side="left", padx=4)
                ttk.Button(r3, text="申请密钥", command=lambda: webbrowser.open(LINKS["deepseek"])).pack(side="left")
                ttk.Label(box, text="密钥只在这次生成时使用，不会保存到任何文件。约 0.5 美元 / 10 万句（以 DeepSeek 当前价格为准）。需要先准备好中文维基（用来取话题词）。",
                          style="S.TLabel").pack(anchor="w")
            else:
                d["files"], d["dialog"] = [], tk.BooleanVar(value=False)
                self.text_list = tk.Listbox(r2, height=3, width=80); self.text_list.pack(side="left")
                c = ttk.Frame(r2); c.pack(side="left", padx=4)
                ttk.Button(c, text="添加 txt", command=self.add_text).pack(fill="x")
                ttk.Button(c, text="移除选中", command=self.remove_text).pack(fill="x")
                ttk.Checkbutton(box, text="这些是对话（聊天记录、问答）", variable=d["dialog"]).pack(anchor="w")
            self.src[k] = d
        ttk.Label(f, text="参考配方（实测，200 句对话评测，万能回答越少越好）：\n"
                          "  · 全配方（老师 0.40 + 维基 0.15 + LCCC 0.45），T10M：万能回答 23.5%，回答不重样 82%（发布的模型）\n"
                          "  · 免费配方（LCCC 0.7 + 维基 0.3），T4M：万能回答 43%，回答不重样 66.5%：能聊，但常说“我也是”\n"
                          "  · 专门主题：你的文本 0.3–0.5，再加 LCCC 或维基保底",
                  style="S.TLabel", justify="left").pack(anchor="w", pady=4)
        v = ttk.LabelFrame(f, text="字表（模型认识的 1024 个字）", padding=6); v.pack(fill="x", pady=4)
        self.vocab = tk.StringVar(value="released")
        ttk.Radiobutton(v, text="用发布模型的字表（推荐：日常中文覆盖约 94%）", value="released", variable=self.vocab).pack(anchor="w")
        ttk.Radiobutton(v, text="按我选的语料新建字表（专业领域、古文、方言等常用字不同时）", value="own", variable=self.vocab).pack(anchor="w")

    def pick_file(self, var, types):
        p = filedialog.askopenfilename(filetypes=types)
        if p: var.set(p)

    def pick_dir(self, var):
        p = filedialog.askdirectory()
        if p: var.set(p)

    def add_text(self):
        for p in filedialog.askopenfilenames(filetypes=[("文本", "*.txt")]):
            self.src["text"]["files"].append(p); self.text_list.insert("end", p)

    def remove_text(self):
        for i in reversed(self.text_list.curselection()):
            self.text_list.delete(i); del self.src["text"]["files"][i]

    def gen_teacher(self):
        d = self.src["teacher"]
        key = d["key"].get().strip()
        if not key:
            messagebox.showinfo("老师语料", "先填 DeepSeek 密钥（点“申请密钥”）"); return
        if not os.path.exists(os.path.join(ROOT, "data", "wiki_raw.json")):
            messagebox.showinfo("老师语料", "生成老师语料要先把中文维基放到 data\\wiki_raw.json（用来取话题词）"); return
        out = d["path"].get()
        env = dict(os.environ, DEEPSEEK_API_KEY=key)
        cmds = []
        if not os.path.exists(os.path.join(ROOT, "data", "teacher", "topics.txt")):
            cmds.append([PY, "data/gen_teacher.py", "--make-topics"])
        cmds.append([PY, "data/gen_teacher.py", "--dir", out, "--sentences", str(d["n"].get()), "--workers", "4"])
        self.run_console(cmds, "老师语料", env=env)

    # ---------------------------------------------------------------- ③ model
    def build_model(self):
        f = self.pages["model"]
        ttk.Label(f, text="模型大小", style="H.TLabel").pack(anchor="w")
        cols = ("size", "params", "blocks", "speed", "ram", "train")
        t = ttk.Treeview(f, columns=cols, show="headings", height=6)
        for c, h, w in zip(cols, ("尺寸", "参数", "命令方块", "游戏里每个字", "游戏内存", "默认步数训练时间（4070S 参考）"), (80, 110, 110, 120, 90, 260)):
            t.heading(c, text=h); t.column(c, width=w, anchor="center")
        for k, lbl, p, b, s, g, ts, st in SIZES:
            t.insert("", "end", iid=k, values=(lbl, f"{p / 1e6:.2f} M", f"{b / 1e4:.0f} 万", f"约 {s} 秒", f"{g} GB", f"{st:,} 步，约 {human(ts * st)}"))
        t.pack(fill="x", pady=4); t.selection_set("t4m"); self.size_tv = t
        t.bind("<<TreeviewSelect>>", lambda e: self.steps.set(dict((s[0], s[7]) for s in SIZES)[t.selection()[0]]))
        g = ttk.Frame(f); g.pack(anchor="w", pady=8)
        self.steps, self.name, self.titlev = tk.IntVar(value=80000), tk.StringVar(), tk.StringVar()
        self.check = tk.BooleanVar(value=False)
        for r, (lbl, w) in enumerate((("训练步数", ttk.Spinbox(g, from_=1000, to=500000, increment=5000, textvariable=self.steps, width=10)),
                                      ("项目名（英文小写、数字、_）", ttk.Entry(g, textvariable=self.name, width=24)),
                                      ("游戏里显示的名字", ttk.Entry(g, textvariable=self.titlev, width=40)))):
            ttk.Label(g, text=lbl).grid(row=r, column=0, sticky="w", pady=3); w.grid(row=r, column=1, sticky="w")
        ttk.Checkbutton(f, text="生成前做逐整数核对（严格解释器逐条执行全部命令；小模型几分钟，T10M 第一次约 3 小时）", variable=self.check).pack(anchor="w")
        ttk.Label(f, text="步数越多越好，但时间成正比；训练随时可以暂停，下次接着练。", style="S.TLabel").pack(anchor="w", pady=6)

    # ---------------------------------------------------------------- ④ run
    def build_run(self):
        f = self.pages["run"]
        b = ttk.Frame(f); b.pack(anchor="w")
        self.btn_start = ttk.Button(b, text="开始 / 继续", command=self.start); self.btn_start.pack(side="left")
        self.btn_pause = ttk.Button(b, text="暂停训练", command=self.pause); self.btn_pause.pack(side="left", padx=6)
        ttk.Button(b, text="打开输出文件夹", command=self.open_out).pack(side="left")
        self.stage_lbl = ttk.Label(f, text="", style="H.TLabel"); self.stage_lbl.pack(anchor="w", pady=8)
        self.bar = ttk.Progressbar(f, length=900, maximum=1000); self.bar.pack(anchor="w")
        self.prog_lbl = ttk.Label(f, text=""); self.prog_lbl.pack(anchor="w", pady=4)
        self.log = tk.Text(f, height=22, width=120, font=("Consolas", 9)); self.log.pack(fill="both", expand=True)

    # ---------------------------------------------------------------- ⑤ chat
    def build_chat(self):
        f = self.pages["chat"]
        ttk.Label(f, text="试聊：在电脑上和训练好的模型说几句（和游戏里逐整数相同的计算，只是不用进游戏）", style="H.TLabel").pack(anchor="w")
        r = ttk.Frame(f); r.pack(anchor="w", pady=6)
        ttk.Label(r, text="模型：").pack(side="left")
        self.chat_src = tk.StringVar(value="project")
        ttk.Radiobutton(r, text="当前项目训练好的模型", value="project", variable=self.chat_src, command=self.chat_reset).pack(side="left")
        ttk.Radiobutton(r, text="发布的 T10M 对话版（models\\d10f）", value="d10f", variable=self.chat_src, command=self.chat_reset).pack(side="left", padx=8)
        self.chat_box = tk.Text(f, height=20, width=110, font=("Microsoft YaHei UI", 11), state="disabled", wrap="char")
        self.chat_box.pack(fill="both", expand=True)
        r2 = ttk.Frame(f); r2.pack(fill="x", pady=6)
        self.chat_in = tk.StringVar()
        e = ttk.Entry(r2, textvariable=self.chat_in, width=70, font=("Microsoft YaHei UI", 11)); e.pack(side="left")
        e.bind("<Return>", lambda ev: self.chat_send())
        ttk.Button(r2, text="发送", command=self.chat_send).pack(side="left", padx=6)
        ttk.Button(r2, text="清空对话", command=self.chat_reset).pack(side="left")
        self.chat_lbl = ttk.Label(f, text="", style="S.TLabel"); self.chat_lbl.pack(anchor="w")
        self.chat_model, self.chat_hist, self.chat_busy = None, [], False

    def chat_say(self, who, text):
        self.chat_box.config(state="normal"); self.chat_box.insert("end", f"{who}：{text}\n"); self.chat_box.config(state="disabled")
        self.chat_box.see("end")

    def chat_reset(self):
        self.chat_hist, self.chat_model = [], None
        self.chat_box.config(state="normal"); self.chat_box.delete("1.0", "end"); self.chat_box.config(state="disabled")
        self.chat_lbl.config(text="")

    def chat_dir(self):
        return os.path.join(ROOT, "models", "d10f") if self.chat_src.get() == "d10f" else os.path.join(self.pdir(), "model")

    def chat_load(self):
        """(spec module, IntModel, vocab, index, dialog flag); the decoding rule as in the game build"""
        from t44k import spec as S
        d = self.chat_dir()
        q = json.load(open(os.path.join(d, "q.json")))
        m = json.load(open(os.path.join(d, "model.json"), encoding="utf-8"))
        vocab = json.load(open(os.path.join(d, "vocab.json"), encoding="utf-8"))["vocab"]
        S.configure(next(k for k, c in S.CONFIGS.items() if c == q["config"]))
        idx = {c: i for i, c in enumerate(vocab)}
        q["decode"] = {"penalty": m["penalty"], "exempt": sorted({idx[c] for c in "，。！？、" if c in idx} | {S.EOS}),
                       "turn_reset": bool(m.get("dialog"))}
        return S, S.IntModel(q), vocab, idx, bool(m.get("dialog"))

    def chat_send(self):
        text = self.chat_in.get().strip()
        if not text or self.chat_busy:
            return
        if not os.path.exists(os.path.join(self.chat_dir(), "model.json")):
            messagebox.showinfo("试聊", "这个项目还没有训练好的模型（要等 ④ 里走到“模型”这一步之后）"); return
        self.chat_in.set(""); self.chat_busy = True
        self.chat_say("我", text); self.chat_lbl.config(text="计算中……（电脑上逐整数模拟，大模型每个字要零点几秒到一秒多）")

        def go():
            try:
                if self.chat_model is None:
                    self.chat_model = self.chat_load()
                S, im, vocab, idx, dialog = self.chat_model
                skipped = "".join(c for c in text if c not in idx)
                ids = [idx[c] for c in text if c in idx]
                if dialog:
                    prompt = [S.BOS] + self.chat_hist + ids + [S.EOS]
                else:
                    prompt = [S.BOS] + ids
                if len(prompt) >= S.T - 1:
                    self.after(0, lambda: (self.chat_say("提示", f"对话已满 {S.T} 个字，请按“清空对话”"), self.chat_lbl.config(text="")))
                    return
                t0 = time.time()
                out = im.generate(prompt, max_new=S.T - len(prompt))
                words = "".join(vocab[t] for t in out if t != S.EOS)
                if dialog:
                    self.chat_hist = prompt[1:] + out
                note = f"（{len(out)} 个字，{time.time() - t0:.1f} 秒" + (f"；字表里没有、被跳过的字：{skipped}" if skipped else "") + "）"
                self.after(0, lambda: (self.chat_say("模型" if dialog else "续写", (text if not dialog else "") + words),
                                       self.chat_lbl.config(text=note)))
            except Exception as e:
                self.after(0, lambda: self.chat_lbl.config(text=f"出错：{e}"))
            finally:
                self.chat_busy = False
        threading.Thread(target=go, daemon=True).start()

    # ---------------------------------------------------------------- projects
    def refresh_projects(self):
        os.makedirs(PROJECTS, exist_ok=True)
        names = sorted(n for n in os.listdir(PROJECTS) if os.path.exists(os.path.join(PROJECTS, n, "project.json")))
        self.proj_box["values"] = names
        if names and not self.proj.get():
            self.proj.set(names[-1]); self.load_project()

    def pdir(self):
        return os.path.join(PROJECTS, self.proj.get())

    def new_project(self):
        self.proj.set(""); self.name.set(""); self.titlev.set("")
        messagebox.showinfo("新建项目", "在 ③ 模型 页填项目名，选好语料和尺寸后点 保存。")

    def collect(self):
        name = self.name.get().strip()
        if not name or not name.replace("_", "a").isalnum() or name != name.lower() or not name[0].isalpha():
            raise ValueError("项目名只能用英文小写字母、数字和 _，并以字母开头（例如 mychat）")
        sources = []
        for k, d in self.src.items():
            if not d["on"].get():
                continue
            s = {"kind": k, "weight": float(d["w"].get())}
            if k == "wiki":
                s["file"] = d["path"].get()
                if not os.path.exists(s["file"]): raise ValueError("找不到中文维基文件：" + s["file"])
            elif k == "lccc":
                s["dir"], s["dedull"] = d["path"].get(), d["dedull"].get()
                if not d["agree"].get(): raise ValueError("使用 LCCC 前请确认它只能用于研究和非商业用途")
                if not os.path.exists(os.path.join(s["dir"], "lccc_base_train.jsonl.gz")): raise ValueError("LCCC 文件夹里没有 lccc_base_train.jsonl.gz")
            elif k == "teacher":
                s["dir"] = d["path"].get()
                if not os.path.exists(os.path.join(s["dir"], "raw.jsonl")): raise ValueError("老师语料文件夹里还没有 raw.jsonl（先生成）")
            else:
                s["files"], s["dialog"] = list(d["files"]), d["dialog"].get()
                if not s["files"]: raise ValueError("“我自己的文本”还没有添加文件")
            sources.append(s)
        if not sources:
            raise ValueError("至少选一种语料")
        size = self.size_tv.selection()[0]
        return {"name": name, "title": self.titlev.get().strip() or f"MCBE Transformer {name}", "config": size,
                "steps": int(self.steps.get()), "batch": 256, "zero_frac": 0.7, "compile": bool(self.env.get("triton")),
                "vocab": self.vocab.get(), "check": bool(self.check.get()), "sources": sources}

    def save_project(self):
        try:
            p = self.collect()
        except ValueError as e:
            messagebox.showwarning("还差一点", str(e)); return None
        d = os.path.join(PROJECTS, p["name"]); os.makedirs(d, exist_ok=True)
        old = os.path.join(d, "project.json")
        if os.path.exists(old):
            o = json.load(open(old, encoding="utf-8"))
            if (o["sources"], o["config"], o.get("vocab")) != (p["sources"], p["config"], p["vocab"]) and \
                    os.path.exists(os.path.join(d, "train")):
                if not messagebox.askyesno("重新开始？", "语料或模型尺寸改了，已有的训练进度会作废（会删除 data / train / model / build）。继续吗？"):
                    return None
                for sub in ("data", "train", "model", "build"):
                    shutil.rmtree(os.path.join(d, sub), ignore_errors=True)
        json.dump(p, open(old, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        self.refresh_projects(); self.proj.set(p["name"])
        return p

    def load_project(self):
        p = json.load(open(os.path.join(self.pdir(), "project.json"), encoding="utf-8"))
        self.name.set(p["name"]); self.titlev.set(p.get("title", "")); self.steps.set(p["steps"])
        self.size_tv.selection_set(p["config"]); self.vocab.set(p.get("vocab", "released")); self.check.set(p.get("check", False))
        for k, d in self.src.items():
            s = next((x for x in p["sources"] if x["kind"] == k), None)
            d["on"].set(s is not None)
            if not s: continue
            d["w"].set(s["weight"])
            if k == "wiki": d["path"].set(s["file"])
            elif k in ("lccc", "teacher"): d["path"].set(s["dir"])
            if k == "lccc": d["agree"].set(True); d["dedull"].set(s.get("dedull", True))
            if k == "text":
                d["files"][:] = s["files"]; d["dialog"].set(s.get("dialog", False))
                self.text_list.delete(0, "end"); [self.text_list.insert("end", x) for x in s["files"]]

    # ---------------------------------------------------------------- run control
    def start(self):
        if self.proc and self.proc.poll() is None:
            return
        p = self.save_project()
        if not p: return
        path = os.path.join(PROJECTS, p["name"], "project.json")
        log = open(os.path.join(PROJECTS, p["name"], "app.log"), "a", encoding="utf-8")
        self.proc = subprocess.Popen([PY, "-u", "-m", "t44k.project", "run", path], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=NOWIN)

    def pause(self):
        if self.proj.get():
            os.makedirs(os.path.join(self.pdir(), "train"), exist_ok=True)
            open(os.path.join(self.pdir(), "train", "STOP"), "w").write("stop")
            self.prog_lbl.config(text="正在暂停：会在下一个整百步存好进度后停下……")

    def open_out(self):
        d = os.path.join(self.pdir(), "build", "输出")
        os.startfile(d if os.path.exists(d) else self.pdir())

    def stage_log(self, d, st):
        """the log of the stage in progress: prepare.log, train.log, or the newest build log"""
        stage = st["stage"] if st else None
        if stage == "语料":
            return os.path.join(d, "prepare.log")
        if stage == "训练":
            return os.path.join(d, "train.log")
        if stage == "生成":
            b = os.path.join(d, "build")
            logs = [os.path.join(b, f) for f in os.listdir(b) if f.endswith(".log")] if os.path.isdir(b) else []
            if logs:
                return max(logs, key=os.path.getmtime)
        return os.path.join(d, "app.log")

    def poll(self):
        try:
            if self.proj.get():
                d = self.pdir()
                st = json.load(open(os.path.join(d, "status.json"), encoding="utf-8")) if os.path.exists(os.path.join(d, "status.json")) else None
                running = self.proc is not None and self.proc.poll() is None
                if st:
                    word = {"running": "进行中", "paused": "已暂停", "done": "完成", "error": "出错"}[st["state"]]
                    i = st["stages"].index(st["stage"])
                    self.stage_lbl.config(text=" → ".join(("【%s】" % s if k == i else s) for k, s in enumerate(st["stages"])) + f"   {word}"
                                          + ("" if running or st["state"] != "running" else "（进程已结束）"))
                pr = os.path.join(d, "train", "progress.json")
                if os.path.exists(pr):
                    p = json.load(open(pr, encoding="utf-8"))
                    self.bar["value"] = 1000 * p["step"] / p["steps"]
                    ev = p.get("last_eval") or {}
                    self.prog_lbl.config(text=f"训练 {p['step']:,} / {p['steps']:,} 步   已用 {human(p['elapsed_s'])}   "
                                              + (f"预计还要 {human(p['eta_s'])}   " if p["status"] == "running" else "")
                                              + (f"验证集下一个字准确率 {ev['val_acc']:.1%}" if ev.get("val_acc") else ""))
                lp = self.stage_log(d, st)
                if lp and os.path.exists(lp):
                    tail = f"[{os.path.basename(lp)}]\n" + open(lp, encoding="utf-8", errors="replace").read()[-6000:]
                    if st and st["state"] == "error":
                        tail += "\n\n出错信息：\n" + st["message"]
                    self.log.delete("1.0", "end"); self.log.insert("end", tail); self.log.see("end")
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
        self.after(2000, self.poll)


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    App().mainloop()
