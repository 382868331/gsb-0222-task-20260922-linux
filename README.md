# 可选停止下的均值证据过程（Mean Evidence Process）

单文件 Python 标准库模块，实现序贯检验中的可下注（test-by-betting）均值证据过程：

- 观测 `x_t ∈ [0,1]`，检验零假设 **H0：条件均值 `E[x_t | F_{t-1}] ≤ m`**，`m ∈ (0,1)`。
- 固定 λ 网格（`0 ≤ λ ≤ 1/m`），每条轨道财富
  `W_λ(t) = ∏_{s≤t} (1 + λ(x_s − m))`，
  均匀混合 `E_t = (1/K) Σ_i W_{λ_i}(t)`。
- 在 H0 下每个因子条件期望 ≤ 1，`W_λ` 与 `E_t` 均为非负上鞅；
  由 Ville 不等式 `P_H0(∃t: E_t ≥ 1/α) ≤ α`，故可在**任意可选停止时刻**下结论。
- 任意时刻 p 值 `p_t = min(1, 1 / max_{s≤t} E_s)`。
- **双侧候选网格版本 `TwoSidedMeanEvidence`**：对调用者给定的有限候选均值网格
  `{μ_i}` 同时维护**上尾/下尾**财富（见下文专节），双侧 α 固定平分，
  返回尚未被任一侧拒绝的候选集合。

## 运行环境与命令

- **Python 3.14.7**（Linux 原生验证；仅使用 3.14 标准库：`math`、`json`、`dataclasses`、`typing`、`fractions`（仅测试））。
- **无需安装任何第三方依赖**，不需要网络、账号、密钥、数据库或 Docker。

```bash
# 必要测试（独立 Fraction 参考 + 契约边界 + 双侧候选网格，共 68 个）
python -m unittest discover -s tests -v

# 固定输入演示（确定性数据，真实计算，实测约 0.01 秒，远小于 90 秒）
python demo.py
```

## 公开接口

```python
from mean_evidence import MeanEvidenceProcess, EvidenceSnapshot

proc = MeanEvidenceProcess(m=0.5, lambdas=[0.0, 0.5, 1.0, 2.0], alpha=0.05)

snap = proc.update(0.9)          # 追加单点，返回 EvidenceSnapshot
snap = proc.extend([0.8, 0.1])   # 批量追加（接受生成器；整批原子）
snap = proc.snapshot()           # 只读当前快照

text = proc.to_json()            # JSON 序列化（另有 to_dict()）
proc2 = MeanEvidenceProcess.from_json(
    text, m=0.5, lambdas=[0.0, 0.5, 1.0, 2.0], alpha=0.05
)                                # 恢复；m/lambdas/alpha 为配置断言
```

`EvidenceSnapshot`（frozen dataclass）字段：

| 字段 | 含义 |
|---|---|
| `t` | 已处理观测数；`t=0` 时 `E_0=1`、`p=1` |
| `evidence` / `log_evidence` | `E_t` 及其自然对数；所有轨道归零时分别为 `0.0` / `-inf` |
| `max_evidence` / `max_log_evidence` | `max_{s≤t} E_s`（运行期峰值，单调不减） |
| `p_value` | `min(1, 1/max_evidence)`，在 log 域以 `exp(−max_log)` 计算 |
| `threshold` | `1/alpha` |
| `stopped` / `stop_time` | 是否已越界及**首次** `E_t ≥ 1/alpha` 的时刻；未越界为 `None` |

其他方法/属性：`proc.m`、`proc.lambdas`（tuple 副本）、`proc.alpha`、`proc.t`、
`proc.log_wealth()`（各轨道 log 财富，归零轨道为 `-inf`）、
`MeanEvidenceProcess.start(...)`（构造器的语义化别名）。

越界只记录 `stop_time`，**不禁止继续追加**；`stop_time` 永远是首次时刻。

## 双侧候选均值网格：`TwoSidedMeanEvidence`

在单侧过程之上提供题面要求的双侧候选网格能力。对调用者给定的**有限候选均值
网格** `mus=[μ_1,…,μ_M]`（每项严格 `0 < μ < 1`）与**上下尾共用的固定 λ 网格**
`lambdas`，每个候选维护两套财富：

- 上尾（检验 H0⁺：`E[x_t|F_{t-1}] ≤ μ_i`）：
  `W⁺_λ(t) = ∏ (1 + λ(x_s − μ_i))`；
- 下尾（检验 H0⁻：`E[x_t|F_{t-1}] ≥ μ_i`）：
  `W⁻_λ(t) = ∏ (1 − λ(x_s − μ_i))`。

**下注仅依赖过去**：第 t 步乘数只含 `x_t` 与既有财富，不用未来信息；
观测允许自适应、可非同分布。

**双侧 α 固定平分**：构造时给定总 α，每侧水平恒为 α/2、Ville 阈值恒为
`2/α`（不做自适应再分配）。上下尾在 H0⁺/H0⁻ 下各为非负上鞅，
联合界（Bonferroni）给出双侧第一类错误 ≤ α；
双侧 p 值 `p = min(1, 2·min(p⁺, p⁻))`，恰在候选被任一侧拒绝时 `p ≤ α`。

**λ 的公共上界**：因子非负要求上尾 `λ ≤ 1/μ_i`、下尾 `λ ≤ 1/(1−μ_i)`，
故网格每项必须满足
`0 ≤ λ ≤ min_i min(1/μ_i, 1/(1−μ_i))`，越界在构造时拒绝。

```python
from mean_evidence import TwoSidedMeanEvidence

proc = TwoSidedMeanEvidence(mus=[0.2, 0.4, 0.6, 0.8],
                            lambdas=[0.0, 1.25], alpha=0.1)
snap = proc.update(0.95)        # 单点；extend(xs) 批量（整批原子）
snap = proc.snapshot()
snap.surviving                  # (0.6, 0.8) —— 尚未被任一侧拒绝的候选 μ
snap.rejected                   # (0.2, 0.4) —— 已拒绝（保持构造顺序）
c = snap.candidates[0]          # 每个候选的 CandidateEvidence
c.upper_evidence / c.lower_evidence          # 当步 E⁺_t / E⁻_t
c.max_upper_log_evidence / max_lower_log_…   # 各侧历史峰值 log
c.upper_p_value / c.lower_p_value            # 每侧原始 p（min(1,1/max E±)）
c.p_value_two_sided                          # min(1, 2·min(p⁺,p⁻))
c.rejected / c.reject_time / c.reject_side   # "upper"/"lower"；同时刻记上尾

proc.candidate_log_wealth(i)    # (上尾各 λ log 财富, 下尾各 λ log 财富)
proc.alpha_per_side             # α/2；proc.threshold == 2/α
text = proc.to_json()           # 另有 to_dict()；from_json/from_dict 恢复
proc2 = TwoSidedMeanEvidence.from_json(text)
```

其他属性：`proc.mus`、`proc.lambdas`（tuple 副本）、`proc.alpha`、`proc.t`、
`TwoSidedMeanEvidence.start(...)`（构造器别名）。拒绝只记录首次时刻与侧别，
**不禁止继续追加**；后续另一侧越界不会改写该候选的首次拒绝信息。

**网格限制（重要）**：结论只对 `mus` 中列出的候选点成立——
模块对网格之外的任何均值不作判断，也不做候选间插值、连续边界求根或
置信区间；候选不被拒绝 ≠ 该 μ 被“接受”，只表示当前证据尚未越过阈值。
λ 网格影响功效但不影响 α 控制；网格在构造时固定，不随数据调整。
本次迭代**不**提供迟到/乱序数据、连续边界求根、自适应策略学习。

### 双侧状态序列化

状态 `kind="two_sided_mean_evidence"`、`version=1`：
`{kind, version, mus, lambdas, alpha, t, upper:[…], lower:[…]}`，
其中上下尾各为 M 个单侧 `version=1` 子状态（下尾以 `m=1−μ`、
观测 `1−x` 的镜像形式存储）。恢复时完整校验种类/版本、长度、
子过程 `m=μ`/`1−μ`、子过程 `alpha` 恰为 `alpha/2`、各子过程 `t` 一致等；
单侧状态不能从双侧入口恢复。不提供配置变更或跨版本迁移路径。

## 输入上限与错误语义

**接口边界（构造参数）**

- `m`：严格 `0 < m < 1` 的有限实数。
- `lambdas`：非空有限序列（list/tuple，非字符串），每项有限且 `0 ≤ λ_i ≤ 1/m`。
  网格在构造时固定；重复值按重复计数（不去重、不静默纠正）。
  未设网格长度上限——它是 O(网格数 × 观测数) 的计算预算边界而非协议限制；
  演示与测试只使用小规模。
- `alpha`：严格 `0 < alpha < 1` 的有限实数。
- 观测 `x_t`：有限且 `0 ≤ x_t ≤ 1`，`int` 或 `float`（接受 `0`、`1` 整数边界）。

**`TwoSidedMeanEvidence` 的附加边界**

- `mus`：非空有限序列（非字符串），每项严格 `0 < μ_i < 1`；重复值按重复计数。
- `lambdas`：每项须满足 `0 ≤ λ ≤ min_i min(1/μ_i, 1/(1−μ_i))`
  （上下尾因子非负的公共上界，错误信息给出具体上界值）。
- 观测校验、`extend` 原子性、NaN/Infinity/bool 拒绝规则与单侧完全一致；
  双侧内部仅以 `1−x` 喂镜像下尾过程，不引入新的数值入口。

**错误类型约定**

- 类型错误（字符串当数值、不可迭代对象、`bool` 等）抛 `TypeError`；
  值域错误（越界、NaN/Infinity、空网格）抛 `ValueError`。
- **所有数值输入拒绝 `NaN`/`Infinity`；整数类参数拒绝 `bool`**
  （Python 中 `bool` 是 `int` 子类，此处显式拒绝 `True/False`）。
- 错误信息带参数名与下标（如 `xs[1]: 必须位于 [0,1]，得到 1.5`），可定位。
- `extend` 先完整校验整批再推进（生成器会先物化为 list），**任一元素非法则整批拒绝，
  过程状态字节级不变**（测试以 `to_dict()` 前后相等验证）。

**序列化与恢复**

- 状态格式 `version=1`：`{version, m, lambdas, alpha, t, log_wealth,
  max_log_evidence, stop_time}`；归零轨道序列化为 JSON `null`。
- `from_dict/from_json` 对状态做完整校验（版本、长度一致、值域、t=0 时峰值必为 0、
  峰值≥当前证据、stop_time 与阈值一致等），损坏状态抛 `ValueError`。
- 恢复时提供的 `m`/`lambdas`/`alpha` 必须与状态中**逐位相等**，否则 `ValueError`：
  **恢复后不能变更 m 或 λ 网格**（alpha 同样锁定）。不提供任何迁移/重配路径。

**搜索预算说明**：本模块没有启发式搜索或可满足性求解，
因此不存在“搜索预算耗尽”与“证明无解”的区分——所有输出均为给定网格上的确定性计算；
参数不合法直接抛错，不存在“找不到解”的返回。

## 浮点误差口径

- **内部全程 log 域**：每步 `log W_i ← log W_i + log(factor)`，
  `factor = fma(λ, x−m, 1.0)`（Python 3.14 `math.fma`，单次舍入）；
  混合值用 `logsumexp` 加 `math.fsum` 计算：
  `log E = h + log((1/K)·Σ exp(logW_i − h))`，`h=max logW_i`。
  长增长流（测试含 5000 步）不会因 `W` 上溢而崩溃；
  `evidence/max_evidence` 超过 double 上界时为 `inf`，此时以 `log_evidence` 为准；
  `p_value` 下溢为 `0.0` 属正常。
- **相对精度**：在测试覆盖范围内（|log E| ≲ 300、K 为小网格、t 至数千步），
  与独立 `Fraction` 精确参考逐点对照，`log E` 与 `log(max E)` 的相对误差
  `< 1e-12·max(1,|log E|)`，p 值绝对误差 `< 1e-12`。
  该口径只描述实测舍入行为，不构成放宽边界判断的容差。
- **边界判断不靠容差**：阈值比较在 log 域按 IEEE-754 计算结果直接判定
  （`log E ≥ log(1/α)`）。典型有理配置（如 m=1/2、λ=2、α=2/3、x=1 恰命中）
  经 fma/舍入逐位相等，测试显式覆盖。若真实值仅以小于舍入误差的量越过阈值，
  可能被判为未越界（反之亦然）——这是有限浮点的固有行为，不用 ε 修补。
- **零因子**：真实因子非负。当 `fma(λ,x−m,1) ≤ 0.0`（恰为 0，
  或极端参数下舍入成微小负值）时该轨道判为归零，log 财富置 `-inf` 且**永久不可复活**；
  这是保守处理（非负财富不应出现负因子）。
- 测试中的独立参考（`tests/test_independent_reference.py`）**不导入被测核心来产生
  期望值**：期望值来自文件内独立的 `Fraction` 有理枚举/递推；序列化往返一致仅作补充。

## 设计取舍

1. **log 域 + fma**：避免财富连乘上溢/下溢；fma 保证因子只舍入一次，
   且在关键边界（`1 − λm = 0`）给出精确 0。
2. **均匀混合、网格固定**：网格是用户的先验选择，本模块不做数据驱动调 λ
   （那会破坏非负上鞅性质）；λ=0 的轨道恒为 1，保证 `E_t > 0`（除非所有轨道归零）。
3. **财富归零即永久剔除**：与数学定义一致，且避免给已死轨道做无意义的 log 累加。
4. **停止是只读事件**：越过阈值后过程仍可追加，便于越过之后继续观察 p 值；
   Ville 界约束的是首次越界概率。
5. **状态只存 log 财富与峰值**：恢复后从财富重算当前 `log E` 并交叉校验峰值，
   不盲信序列化的派生量。
6. **双侧复用单侧（组合而非复制算法）**：每个候选的下尾过程就是
   `m=1−μ`、观测 `1−x` 的单侧过程（`1+λ((1−x)−(1−μ)) ≡ 1−λ(x−μ)`），
   因此上下尾共用同一套经过验证的 log 域/fma/归零逻辑；α 固定平分，
   不做任何数据驱动的权重调整（那会破坏上鞅与 α 保证）。

## 已知限制 / 范围声明

- 证据**只针对条件均值上界 H0：`E[x_t | F_{t-1}] ≤ m`**。
  该保证允许观测自适应、可非同分布，**不要求独立性**；相应地，模块不对独立性、
  分布形态、方差、分位数或相关性作任何额外断言，λ 网格选择影响功效但不影响第一类错误控制。
- 要求 `x_t ∈ [0,1]` 与 `0 ≤ λ ≤ 1/m`，这是因子非负（Ville 适用）的前提；
  无界观测或其他参数化（如 e-process 的其他混合测度）不在本题契约内。
- 双侧候选网格（`TwoSidedMeanEvidence`）额外限定：结论只覆盖 `mus` 中列出的
  有限候选点，**不做**网格外推断、候选间插值、连续 μ 上的边界求根或置信集；
  “未拒绝”不等于接受。本次迭代也**不**处理迟到/乱序数据与自适应策略学习；
  双侧 α 只支持固定平分（每侧 α/2）。
- 不实现流式网格扩展、连续 λ 积分、多重检验校正，也不宣称兼容任何行业标准协议；
  这只是本题规定的有限数学子集。
- 线程安全：对象无内置锁；多线程使用请自行同步。
- 状态 JSON 为人工可读格式，但仅支持 `version=1`（双侧另有 `kind` 标识），
  跨版本迁移未实现。

## 文件

- `mean_evidence.py` — 核心模块（无第三方依赖）：单侧 `MeanEvidenceProcess`
  与双侧候选网格 `TwoSidedMeanEvidence`、快照 dataclass。
- `tests/test_independent_reference.py` — 单侧独立 Fraction 参考：两点 Bernoulli 全路径枚举
  验证 H0 下 `E[W_λ] ≤ 1`、Ville 命中质量 ≤ α，以及与核心的逐点对照。
- `tests/test_two_sided.py` — 双侧独立 Fraction 参考（期望值不来自被测核心）：
  手算全 1/全 0、退化 λ=0、财富双侧归零、候选集合收缩、Bernoulli 命中质量
  （含 α 界紧的情形）、构造/观测校验与批量原子性、序列化损坏状态拒绝。
- `tests/test_contract.py` — 0/1 边界、零因子、非法 λ、NaN/bool 拒绝、
  批量原子性、状态恢复/损坏/配置锁定、长流不溢出等契约测试。
- `demo.py` — 固定确定性输入：单侧正常停止/不拒绝、三类实际异常边界、
  序列化恢复一致性，以及双侧候选网格的正常收缩、退化下注、全 0/全 1、
  λ 上界与候选边界、批量错误的原子性（场景 E–F）。
