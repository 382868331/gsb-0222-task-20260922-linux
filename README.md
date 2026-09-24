# 可选停止下的均值证据过程（Mean Evidence Process）

单文件 Python 标准库模块，实现序贯检验中的可下注（test-by-betting）均值证据过程，
以及建立在它之上的**有限均值网格双侧置信集**：

- 观测 `x_t ∈ [0,1]`，检验零假设 **H0：条件均值 `E[x_t | F_{t-1}] ≤ m`**，`m ∈ (0,1)`。
- 固定 λ 网格（`0 ≤ λ ≤ 1/m`），每条轨道财富
  `W_λ(t) = ∏_{s≤t} (1 + λ(x_s − m))`，
  均匀混合 `E_t = (1/K) Σ_i W_{λ_i}(t)`。
- 在 H0 下每个因子条件期望 ≤ 1，`W_λ` 与 `E_t` 均为非负上鞅；
  由 Ville 不等式 `P_H0(∃t: E_t ≥ 1/α) ≤ α`，故可在**任意可选停止时刻**下结论。
- 任意时刻 p 值 `p_t = min(1, 1 / max_{s≤t} E_s)`。
- 双侧扩展：对调用者给定的有限候选均值网格 `{m_j}`，每个候选同时维护
  上尾（H0: μ ≤ m_j）与下尾（H0: μ ≥ m_j）两条财富，双侧 α 固定平分为每侧 α/2，
  返回**尚未被拒绝的候选集合**（序贯置信集）。

## 运行环境与命令

- **Python 3.14.7**（Linux 原生验证；仅使用 3.14 标准库：`math`、`json`、`dataclasses`、`typing`、`fractions`（仅测试））。
- **无需安装任何第三方依赖**，不需要网络、账号、密钥、数据库或 Docker。

```bash
# 必要测试（独立 Fraction 参考 + 契约边界 + 网格置信集测试，共 70 个）
python -m unittest discover -s tests -v

# 固定输入演示（确定性数据，真实计算，实测约 0.05 秒，远小于 90 秒）
python demo.py
```

## 公开接口

### 单侧证据过程 `MeanEvidenceProcess`

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

### 有限均值网格双侧置信集 `MeanGridConfidenceSet`

```python
from mean_evidence import MeanGridConfidenceSet, ConfidenceSnapshot

cs = MeanGridConfidenceSet(
    candidates=[0.2, 0.35, 0.5, 0.65, 0.8],   # 调用者给定的有限候选均值网格
    lambdas=[0.0, 0.5, 1.0, 1.25],            # 两侧共享的固定 λ 网格
    alpha=0.1,                                # 双侧总水平，固定平分为每侧 0.05
)

snap = cs.update(0.7)          # 追加单点，返回 ConfidenceSnapshot
snap = cs.extend([0.6, 0.8])   # 批量追加（整批原子）
cs.confidence_set()            # 当前尚未被拒绝的候选 tuple
snap2 = cs.snapshot()

text = cs.to_json()            # 序列化（version=1 网格状态格式）
cs2 = MeanGridConfidenceSet.from_json(
    text, candidates=[0.2, 0.35, 0.5, 0.65, 0.8],
    lambdas=[0.0, 0.5, 1.0, 1.25], alpha=0.1,
)                              # 恢复；candidates/lambdas/alpha 为配置断言
```

**数学定义**：对每个候选 `m_j` 同时维护两条财富过程——

- 上尾：检验 H0⁺：`E[x_t|F_{t-1}] ≤ m_j`，因子 `1 + λ(x_t − m_j)`；
- 下尾：检验 H0⁻：`E[x_t|F_{t-1}] ≥ m_j`，因子 `1 + λ(m_j − x_t)`
  （实现上等价于对 `y_t = 1 − x_t` 跑 `m' = 1 − m_j` 的上尾过程，复用同一上鞅实现）。

**下注仅依赖过去**：下注策略是构造时固定的常数 λ 网格——常数不依赖任何观测，
自动满足关于 `F_{t-1}` 可预测的要求，两条财富在各自零假设下都是非负上鞅。

**双侧 α 固定平分**：每侧水平 `alpha/2`，每侧拒绝阈值 `2/alpha`。
候选 `m_j` 在任一侧混合财富**首次** `≥ 2/alpha` 时被拒绝，拒绝不可逆
（记录首次越界时刻，之后即使财富回落也不恢复）。由 Ville 不等式与并集界：
若真实均值 μ 在网格中，则 `P(μ 曾被拒绝) ≤ alpha/2 + alpha/2 = alpha`。

`ConfidenceSnapshot`（frozen dataclass）字段：

| 字段 | 含义 |
|---|---|
| `t` | 已处理观测数 |
| `alpha` / `alpha_side` | 双侧总水平 / 每侧水平（`alpha/2`） |
| `threshold` | 每侧拒绝阈值 `2/alpha` |
| `candidates` | 完整候选网格（构造时固定） |
| `rejected` / `confidence_set` | 已拒绝 / 尚未拒绝的候选（均按网格顺序） |
| `upper_log_evidence` / `lower_log_evidence` | 各候选两侧 log 证据，与 `candidates` 对齐；财富归零为 `-inf` |
| `rejection_times` | 各候选首次被拒绝时刻（未拒绝为 `None`），与 `candidates` 对齐 |

**网格限制（务必注意）**：置信集永远是调用者给定网格的子集——它表达的是
“网格内哪些候选尚未被拒绝”，对网格外的均值不作任何陈述。若真实均值不在
网格中，置信集**可能合法地收缩为空集**；空集的含义是“网格内没有与数据相容
的候选”，应加密网格或换用连续方法（本模块不提供）。覆盖保证
`P(μ 被保留) ≥ 1 − alpha` 只在 `μ ∈ 网格` 时成立。

## 输入上限与错误语义

**接口边界（构造参数）**

- `m`：严格 `0 < m < 1` 的有限实数。
- `lambdas`：非空有限序列（list/tuple，非字符串），每项有限。
  - `MeanEvidenceProcess`：`0 ≤ λ_i ≤ 1/m`；
  - `MeanGridConfidenceSet`：两侧共享同一网格，要求
    `0 ≤ λ_i ≤ min_j min(1/m_j, 1/(1−m_j))`（保证每个候选两侧的因子都非负）。
  网格在构造时固定；重复值按重复计数（不去重、不静默纠正）。
  未设网格长度上限——它是 O(网格数 × 观测数) 的计算预算边界而非协议限制；
  演示与测试只使用小规模。
- `candidates`（仅 `MeanGridConfidenceSet`）：非空有限序列，每项严格 `0 < m_j < 1`，
  **不允许重复**（重复候选抛 `ValueError`）。
- `alpha`：严格 `0 < alpha < 1` 的有限实数。
- 观测 `x_t`：有限且 `0 ≤ x_t ≤ 1`，`int` 或 `float`（接受 `0`、`1` 整数边界）。

**错误类型约定**

- 类型错误（字符串当数值、不可迭代对象、`bool` 等）抛 `TypeError`；
  值域错误（越界、NaN/Infinity、空网格、重复候选）抛 `ValueError`。
- **所有数值输入拒绝 `NaN`/`Infinity`；整数类参数拒绝 `bool`**
  （Python 中 `bool` 是 `int` 子类，此处显式拒绝 `True/False`）。
- 错误信息带参数名与下标（如 `xs[1]: 必须位于 [0,1]，得到 1.5`），可定位。
- `extend` 先完整校验整批再推进（生成器会先物化为 list），**任一元素非法则整批拒绝，
  过程状态字节级不变**（测试以 `to_dict()` 前后相等验证）。

**序列化与恢复**

- 单侧过程状态格式 `version=1`：`{version, m, lambdas, alpha, t, log_wealth,
  max_log_evidence, stop_time}`；归零轨道序列化为 JSON `null`。
- 网格置信集状态格式 `version=1`（独立常量 `GRID_STATE_VERSION`）：
  `{version, candidates, lambdas, alpha, t, upper, lower}`，其中 `upper`/`lower`
  为逐候选的单侧过程状态列表。恢复时交叉校验：上尾子过程的 `m` 必须等于对应
  候选、下尾必须等于 `1 − 候选`、子过程 λ 网格与 `alpha` 必须与顶层一致、
  两侧 `t` 必须相等——不盲信序列化内容。
- `from_dict/from_json` 对状态做完整校验（版本、长度一致、值域、t=0 时峰值必为 0、
  峰值≥当前证据、stop_time 与阈值一致等），损坏状态抛 `ValueError`。
- 恢复时提供的 `m`/`candidates`/`lambdas`/`alpha` 必须与状态中**逐位相等**，
  否则 `ValueError`：**恢复后不能变更 m、候选网格或 λ 网格**（alpha 同样锁定）。
  不提供任何迁移/重配路径。

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
- **下尾变换**：`MeanGridConfidenceSet` 的下尾以 `y = 1 − x`、`m' = 1 − m` 计算，
  与直接算 `1 + λ(m − x)` 在数学上恒等；非二进制有理数输入下两者可差约 1 ulp，
  对照测试以 `1e-12` 相对容差覆盖，恰命中边界的用例全部使用二进制有理参数
  （0.5、2 的幂），不存在舍入歧义。
- 测试中的独立参考（`tests/test_independent_reference.py`、
  `tests/test_grid_confidence.py`）**不导入被测核心来产生期望值**：
  期望值来自文件内独立的 `Fraction` 有理枚举/递推；序列化往返一致仅作补充。

## 设计取舍

1. **log 域 + fma**：避免财富连乘上溢/下溢；fma 保证因子只舍入一次，
   且在关键边界（`1 − λm = 0`）给出精确 0。
2. **均匀混合、网格固定**：网格是用户的先验选择，本模块不做数据驱动调 λ
   （那会破坏非负上鞅性质）；λ=0 的轨道恒为 1，保证 `E_t > 0`（除非所有轨道归零）。
3. **财富归零即永久剔除**：与数学定义一致，且避免给已死轨道做无意义的 log 累加。
4. **停止是只读事件**：越过阈值后过程仍可追加，便于越过之后继续观察 p 值；
   Ville 界约束的是首次越界概率。同理，网格置信集中候选一旦被拒绝即永久拒绝。
5. **状态只存 log 财富与峰值**：恢复后从财富重算当前 `log E` 并交叉校验峰值，
   不盲信序列化的派生量。
6. **双侧复用单侧实现**：下尾通过 `y = 1 − x` 变换复用同一个经过完整测试的
   上鞅过程，避免维护第二份 log 域递推；α 平分用并集界即可保证总水平，
   不做数据驱动的 α 分配（那会破坏可预测性）。

## 已知限制 / 范围声明

- 证据**只针对条件均值界 H0：`E[x_t | F_{t-1}] ≤ m`（及其下尾对偶）**。
  该保证允许观测自适应、可非同分布，**不要求独立性**；相应地，模块不对独立性、
  分布形态、方差、分位数或相关性作任何额外断言，λ 网格选择影响功效但不影响第一类错误控制。
- 要求 `x_t ∈ [0,1]` 与相应的 λ 上界，这是因子非负（Ville 适用）的前提；
  无界观测或其他参数化（如 e-process 的其他混合测度）不在本题契约内。
- 置信集是**有限网格上的离散集合**，不是连续区间：网格外的均值不在表达范围内，
  真值不在网格中时置信集可能合法地为空；连续边界求根、网格自适应加密、
  迟到/乱序数据、自适应下注策略学习均不在本次交付范围。
- 不实现流式网格扩展、连续 λ 积分、多重检验校正，也不宣称兼容任何行业标准协议；
  这只是本题规定的有限数学子集。
- 线程安全：对象无内置锁；多线程使用请自行同步。
- 状态 JSON 为人工可读格式，但仅支持 `version=1`，跨版本迁移未实现。

## 文件

- `mean_evidence.py` — 核心模块（无第三方依赖）：单侧 `MeanEvidenceProcess`
  与双侧网格置信集 `MeanGridConfidenceSet`。
- `tests/test_independent_reference.py` — 独立 Fraction 参考：两点 Bernoulli 全路径枚举
  验证 H0 下 `E[W_λ] ≤ 1`、Ville 命中质量 ≤ α，以及与核心的逐点对照。
- `tests/test_contract.py` — 0/1 边界、零因子、非法 λ、NaN/bool 拒绝、
  批量原子性、状态恢复/损坏/配置锁定、长流不溢出等契约测试。
- `tests/test_grid_confidence.py` — 网格置信集测试：独立 Fraction 双侧参考逐点对照、
  固定下注退化（λ 全 0 永不拒绝）、财富归零不可复活、全 0/全 1 序列的拒绝时刻
  手算锚点、α 平分与恰命中阈值、Bernoulli 全路径枚举验证真候选拒绝质量 ≤ α、
  非法输入、批量原子性、序列化/恢复/配置锁定。
- `demo.py` — 固定确定性输入：正常停止、正常不拒绝、三类实际触发的异常边界、
  序列化恢复一致性，以及网格置信集的正常收缩、空集/财富归零边界与非法输入异常。
