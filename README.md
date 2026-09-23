# 可选停止下的均值证据过程（mean_evidence）

一个零第三方依赖的 Python 标准库模块：对取值于 `[0,1]` 的序贯观测
`x_t`，在零假设 **条件均值 `E[x_t | F_{t-1}] <= m`** 下构造可在任意
时刻停止检验的财富/证据过程，并返回首次拒绝时刻与 anytime p 值。

- Python 版本：**3.14.7**（仅使用 3.10+ 通用语法，标准库）
- 依赖：**无**。不安装任何第三方包，不需要网络、账号、密钥、数据库、
  中间件、云 API 或 Docker。
- 核心、测试、演示全部离线可复现；示例数据由仓库内固定种子的
  `random.Random` 确定性生成。

## 快速命令

```bash
python -m unittest discover -s tests -v   # 34 个测试
python demo.py                            # 端到端演示，实测 < 1 秒
```

## 数学定义与适用范围

给定固定的 λ 网格 `0 <= λ_k <= 1/m`，定义

```
W_{λ,t} = Π_{s=1..t} (1 + λ (x_s - m))          # 单项财富，W_0 = 1
E_t     = (1/K) Σ_k W_{λ_k,t}                    # 对网格均匀混合，E_0 = 1
```

零假设下，对每个 λ，

```
E[1 + λ(x_t - m) | F_{t-1}] = 1 + λ(E[x_t|F_{t-1}] - m) <= 1,
```

故 `E[W_{λ,t}] <= 1`、`E[E_t] <= 1`；由 Ville 不等式，对任意（可测）
停时 τ，

```
P(存在 t: E_t >= 1/α) <= α,
```

因此可以**边看边停（optional stopping）**，在首次 `E_t >= 1/α` 时
拒绝零假设而不膨胀一类错误。anytime p 值为

```
p_t = min(1, 1 / max_{s<=t} E_s)。
```

**适用范围（本库只负责这一结论）**：零假设必须逐期成立——只需要
条件均值不等式，**不要求观测之间相互独立**（鞅/上鞅论证只依赖条件
期望）。本库不对以下内容作任何承诺：分布形态、条件方差、备择假设
下的功效最优性、网格之外 λ 的行为，或把证据推广为“独立性成立”等
其他命题。

## 接口

```python
from mean_evidence import (
    MeanEvidenceProcess,   # 有状态过程
    AnalysisResult,        # analyze() 的只读结果
    analyze,               # 一次性便捷分析
    EvidenceError,                 # 本库错误基类（也是 ValueError）
    InvalidParameterError,         # m / grid / alpha / 恢复参数非法
    InvalidObservationError,       # 观测 x 非法或越界
    StateMismatchError,            # 恢复时 m/grid/alpha 与状态不一致
    SerializationError,            # 序列化数据损坏/版本不符
)

proc = MeanEvidenceProcess(m=0.5, lambda_grid=[0.0, 0.5, 1.0, 2.0], alpha=0.05)
proc.append(1)                       # 追加单个观测，返回 log E_t
proc.extend([0, 1, 1])               # 批量追加（失败整体回滚）
proc.evidence          # float，当前 E_t
proc.log_evidence      # float，当前 log E_t（全部归零时 -inf）
proc.log_wealth        # tuple[float,...]，各 λ 的 log W（归零项为 -inf）
proc.max_log_evidence  # 历史最大 log E
proc.p_value           # float，anytime p = min(1, exp(-max log E))
proc.stop_time         # int | None，首次 E_t >= 1/alpha 的时刻（从 1 起）
proc.rejected          # bool

# 序列化（恢复时必须显式重述 m / 网格 / alpha，逐一严格相等，否则拒绝）
blob = proc.to_json()
proc2 = MeanEvidenceProcess.from_json(blob, m=0.5, lambda_grid=[...], alpha=0.05)
d = proc.to_dict(); MeanEvidenceProcess.from_dict(d, m=..., lambda_grid=..., alpha=...)

# 一次性
r = analyze(0.5, [0.0, 1.0, 2.0], observations, 0.05)
# r.t / r.log_evidence / r.p_value / r.stop_time / r.alpha / r.rejected
```

## 输入契约（接口边界）

| 参数 | 约束 |
|---|---|
| `m` | `0 < m < 1` 的有限实数 |
| `lambda_grid` | 非空有限数值序列；每项有限且 `0 <= λ <= 1/m`；构造后不可更改 |
| `alpha` | `0 < alpha < 1` 的有限实数；拒绝阈值 `1/alpha` |
| 观测 `x` | 有限实数且 `0 <= x <= 1`（边界 0、1 合法；`int` 合法） |

通用错误语义：

- 所有数值输入拒绝 `NaN` / `+Infinity` / `-Infinity`；
- `bool` 不冒充数值或整数（`True`/`False` 一律拒绝，即使 Python 中
  `bool` 是 `int` 子类）；
- 字符串等非数值类型拒绝。错误消息携带具体参数名（如
  `lambda_grid[2]`、`xs[3]`），**非法输入不静默纠正**；
- `extend`/`append` 在任何状态更新前完成整批校验；更新本身在快照上
  进行，失败即整体回滚，不留下未说明的部分变更；
- 恢复状态必须显式传入且与状态记录的 `m`、`lambda_grid`、`alpha`
  **严格相等**；不允许借恢复更换 m 或网格。JSON 中出现 `NaN`/
  `Infinity` 常量、版本不符、字段缺失、长度不符或内部量自相矛盾
  （如已越阈值却无 `stop_time`）都抛 `SerializationError`。

本模块没有“搜索预算耗尽”这类非确定性失败：没有数值搜索，因此
不存在预算耗尽与“证明无解”相混淆的路径——输入不合法即明确报错，
计算不会静默给出无解结论。

## 设计取舍

1. **全程 log 域**。保存每项 `log W` 的累加和，用 logsumexp 求混合：
   `log E = logsumexp_k(log W_k) - log K`；阈值比较用
   `log E >= -log α`。财富即便超过 `float.max`，停止时刻与 p 值判定
   也不受影响（`evidence` 直流量此时返回 `inf`）。
2. **零因子显式处理**。因子恰为 0（含 `λ=1/m` 且 `x=0`）时该项财富
   永久标记死亡（`log W = -inf`，序列化用 `null`），之后不再参与
   混合，也不会“复活”；全部财富死亡时 `E_t = 0`。
3. **等号归拒绝**：`E_t >= 1/α` 即停（阈值含等号）。
4. **p 值锚定运行最大值**，先冲高后回落不会让 p 值反弹；历史最大值
   含 `E_0 = 1`，故无证据时 `p_t = 1`（天然实现 `min(1, ·)`）。
5. **网格固定、均匀权重**：网格是接口身份的一部分，恢复时逐位核对；
   不提供自适应网格或其他混合测度（避免在本题契约之外宣称功效
   性质）。
6. **反序列化做结构与内部一致性校验**（长度、非负步数、`log W` 不超
   理论上界 `t·log(1/m)`、运行最大值与阈值/停止标志相互对得上），
   防止篡改状态驱动后续计算。

## 浮点误差口径

- 停止判定与 p 值基于 log 域比较。`log W` 为每步
  `log(1+λ(x-m))`（因子为正时按 IEEE-754 正确舍入）的简单累加；
  `log E` 经一次 logsumexp（减法对齐 + 求和 + 一次 log）得到。
  在本库面向的短/中序列（测试到数百步量级）上，`log E` 与精确值的
  偏差为 O(t)·ulp 量级；测试以 `rel_tol=1e-14` 与按定义独立重算的
  参考值核对（t 较小时即机器精度附近），**不用宽容差掩盖边界**：
  阈值等号情形用二进制精确可表示的构造（`E_t = 2` 恰好命中
  `1/α = 2`）逐位断言。
- 边界舍入：`λ = 1/m` 与 `x = 0` 时，`λ·m` 可能因舍入使因子成为
  绝对值 ≤ `8·ulp(1) ≈ 1.78e-15` 的微小负值；数学上该因子本就是 0，
  此幅度内按零因子处理（财富归零）。超出该幅度的负因子意味着内部
  不变量被破坏，直接报错而不是静默夹断。
- `p_value = exp(-max_log_evidence)`：证据极大时 p 正确下溢为
  `0.0`（舍入结果，不另设容差）；`evidence = exp(log_evidence)`
  上溢时为 `inf`，仅影响该直流量，不影响判定。
- 上述误差只影响数值舍入，不影响 Ville 不等式的数学保证；不宣称
  完整行业标准兼容，本库只实现本题契约的有限数学子集。

## 已知限制

- 仅支持 `x ∈ [0,1]`、`m ∈ (0,1)` 的有界单边均值检验；不支持
  无界变量、双侧检验或备择功效优化。
- 网格为用户给定的固定有限序列；不内置网格选择，也不保证任意网格
  的备择功效。
- 状态为 Python JSON 字典/字符串；版本号当前为 `1`，不承诺跨版本
  格式兼容。
- 不做并发保护：一个进程实例不是线程安全对象。
- 状态恢复要求参数浮点**严格相等**（不做“近似同一网格”的模糊匹配）。

## 目录

```
mean_evidence/__init__.py     # 公开接口
mean_evidence/errors.py       # 错误类型
mean_evidence/validation.py   # 输入校验（拒 NaN/Inf/bool）
mean_evidence/process.py      # 核心：log 域证据过程、零因子、序列化
tests/test_core.py            # 行为/边界/错误语义（独立按定义重算参考值）
tests/test_bernoulli_enumeration.py  # Bernoulli 全路径 Fraction 精确枚举
tests/test_serialization.py   # 状态往返、身份锁定、损坏状态
demo.py                       # 固定输入演示
```

测试中的 Bernoulli 枚举使用 `fractions.Fraction` 对全部 `2^n` 条路径
精确计算，**不调用被测核心**来产生期望值；核心仅用于停止时刻的
一致性交叉核对。
