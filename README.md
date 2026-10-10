# MCBE Transformer

**在《我的世界》基岩版里，只用原版命令方块和记分板，跑一个真正会生成文字的 Transformer 语言模型。**

模型的每一步计算都在游戏里完成：
- 把字变成向量；
- 6 层注意力和前馈网络；
- 选出下一个字；
- 把生成的字接回输入、继续生成。

运行时不需要 MOD、行为包脚本、外部程序或网络。世界搭好后可以移除全部附加包，照样能用。游戏里算出的每一个整数，都和 Python 写的整数规范逐个一致。

[English summary](#english-summary)

---

## 现在能做到什么

| | T10M 对话版（本仓库发布） |
|---|---|
| 参数 | 966 万（6 层，宽 384，8 个注意力头，前馈 1152） |
| 词表 / 上下文 | 1024 个常用汉字和标点 / 64 个字 |
| 游戏里的规模 | 3,107,262 个命令方块，一条命令链 |
| 速度 | 每个字 1 个 tick，约 11 秒（i5-13400F） |
| 内存 | 游戏进程约 7–9 GB |
| 正确性 | 用发布的行为包在新世界里安装后，实机 4 段对话 131 次前向和 Python 整数规范逐整数一致；310 万个方块在游戏里逐个核对无误 |
| 输入 | 世界里的 26 键拼音键盘（按钮 + 命令方块），能多轮对话 |

它能做的是**很短的日常闲聊**：打招呼、节日祝福、简单的是非问、情绪回应。比如（T10M 离线评测）：
- "中秋快乐" → "你也快乐"
- "不减肥了？" → "不减了"

它**不懂知识，也不会算数**，长句子容易答非所问。有时候会整句打转，比如"梦里有个梦，梦里有个梦……"。966 万参数，训练数据是网上的短对话，这就是它的水平。

## 三种用法

### 1. 直接玩（最简单）

1. 从 [Releases](../../releases) 下载 `.mcworld`，双击导入。
2. 进世界后，先按键盘最上排右侧的 **开机**。首次加载要几分钟，游戏会卡住，属于正常现象。
3. 按 **初始化**。
4. 用字母键打拼音，用数字键选字，按 **发送**。

详见 [部署指南](docs/部署指南.md)。

### 2. 装进你自己的世界（不需要任何开发工具）

1. 下载 `d10f.mcpack`（国际版），或 `d10f_netease.zip`（网易 MC Studio 组件）。
2. 新建一个**超平坦**世界，开启作弊，挂上这个行为包。
3. 在聊天栏输入 `/function mct/install`。模型分 15 批自动加载，每批游戏会卡几秒到几十秒。
4. 看到"模型已装好"后，站到空地上**面朝南**，输入 `/function mct/keyboard`。
5. 按 **初始化**。

装好以后可以移除这个行为包，世界里只剩原版方块。

### 3. 自己训练、自己部署

```bash
pip install -r requirements.txt
python -m t44k.pipeline build models/d10f out/d10f --check
python -m t44k.pipeline train d10f --dry-run
```

- 第二行：把发布的权重编译成行为包，并核对整数一致。
- 第三行：列出重新训练这个模型的完整步骤。

数据从哪里下载、要花多少钱和时间、怎么换成更小的模型，都在 [训练指南](docs/训练指南.md) 里。

## 它是怎么工作的

| 问题 | 做法 |
|---|---|
| 记分板只有 32 位整数 | 网络本身就定义成整数运算：[`t44k/spec.py`](t44k/spec.py) 是唯一真相，训练时在同样的整数单位里做量化感知训练 |
| 一条命令能做的事很少 | 权重是三值（−1/0/+1），一个非零权重编译成一条 `scoreboard players operation … += / -=`，零权重不生成命令 |
| 除法和负数 | 一律写成"加偏移 → 整除 → 减偏移"，和平台的取整方式无关 |
| RMSNorm、softmax | 查找表 |
| 一个字要算 200 万条命令 | 整次前向放进**一条链、一个 tick**（`gamerule maxcommandchainlength 4000000`） |
| 同一 tick 改太多不同的假玩家名会极慢 | 寄存器映射成"256 个名字 × 很多记分项"（[`regmap.py`](t44k/regmap.py)），提速 4.8 倍 |
| 命令太多 | 三值矩阵公共子式消除（[`cse.py`](t44k/cse.py)），矩阵乘命令少 40%；嵌入表 3 个数打包进一个分数 |
| 循环链头断电后还会多跑 1–2 tick | 链尾把第 2 个方块换成石头，开始时用 `clone` 换回来 |
| 网易会静默吞掉含敏感词的命令 | 构建时检查（[`netease_lint.py`](netease_lint.py)），上机前把所有名字在游戏里试写一遍 |

验证分四层，每一层都必须逐整数通过：
1. 整数规范；
2. 严格命令解释器逐条执行真实命令文本（两种除法语义）；
3. 逐 tick 世界模拟；
4. 实机比对。

完整原理见 [技术手册](docs/HANDBOOK.md) 和 [引擎说明](docs/T44K_引擎.md)。

## 走过的路

我们在 1000 万参数附近摸到了这条路线的边界：
- **内存**：约 2.5 KB/命令方块，310 万块就是 9 GB 多。
- **时间**：约 5 µs/条命令，每字 11 秒。

为了越过它试过很多办法，大部分没有成功：
- 实体选择器并行；
- 结构换页；
- 分段扫描；
- 跨层共享权重；
- 更稀疏的权重；
- 方块记忆里的 n-gram 统计。

这些"走不通"的结果和测量数据都在 [实验结论](docs/实验结论.md) 里，希望能帮后来的人少走弯路。

## 目录

```
t44k/            引擎：整数规范、编译器、布局、核对、模拟、训练、拼音键盘、行为包生成、一键流水线
data/            语料处理脚本（维基、DeepSeek 老师语料、LCCC 对话、去万能回答、问答挖掘）
data/mixd/       发布模型的字表
models/d10f/     发布的 T10M 对话版权重（仅限研究，见 DATA_AND_MODEL.md）
vm.py            严格命令解释器（离线核对用）
nbt.py           小端 NBT 读写
netease_lint.py  网易敏感词检查
docs/            技术手册、部署 / 训练指南、实验记录
```

## 许可

- **代码**：MIT，见 [LICENSE](LICENSE)。
- **模型权重和世界存档**：**仅限研究、非商业用途**。原因是训练数据含 LCCC，其作者声明仅供研究使用。详见 [DATA_AND_MODEL.md](DATA_AND_MODEL.md)。
- 本仓库不包含任何训练语料，脚本只负责处理你自己下载的数据。

## 致谢

- **LCCC**（清华大学 CoAI 小组）：中文短对话语料。
- **中文维基百科**：贡献者的文字，CC BY-SA 3.0，经 `0xDing/wikipedia-cn-20230720-filtered` 整理。
- **DeepSeek**：开放平台 API，生成老师语料。
- **pypinyin**：拼音键盘的读音表。
- **mod-debug-bridge**（sevenstars0，BSD-3）：开发期在网易版游戏里执行 Python 的调试桥（未包含在本仓库）。

---

## English summary

A real generative Transformer language model that runs **entirely on vanilla command blocks and scoreboards** in Minecraft Bedrock Edition (tested on NetEase 1.21.120):
- every token's forward pass — embedding, 6 attention/FFN layers, argmax and feeding the token back — happens in-game;
- the world keeps working with all add-ons removed;
- every integer matches the Python reference spec exactly.

The released model has 9.66M parameters (ternary weights, 1024-character Chinese vocabulary, context 64). It runs as 3.1M command blocks in one chain, one token per tick, at about 11 s per character and 7–9 GB of game memory. It manages short everyday Chinese small talk, with no knowledge and no arithmetic.

- **Play**: import the `.mcworld` from Releases.
- **Install into your own world**: in a new flat world, add `d10f.mcpack` and run `/function mct/install`, then `/function mct/keyboard`.
- **Train / compile your own**: `python -m t44k.pipeline`.

Code: MIT. Weights and world: research / non-commercial use only, because the training data includes LCCC (see `DATA_AND_MODEL.md`).
