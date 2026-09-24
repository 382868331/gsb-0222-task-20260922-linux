"""可选停止下的均值证据过程（mean evidence process with optional stopping）。

数学设定（见 README 详述）：

- 观测 x_t ∈ [0,1]，待检验（零假设）条件均值 E[x_t | F_{t-1}] ≤ m，m ∈ (0,1)。
- 固定网格 λ_i ∈ [0, 1/m]，每条轨道财富
      W_λ(t) = ∏_{s≤t} (1 + λ(x_s - m))。
  因 0 ≤ λ ≤ 1/m 且 x_s ≥ 0，各因子非负；在 H0 下因子条件期望 ≤ 1，
  故 W_λ 是非负上鞅，均匀混合 E_t = (1/K)Σ_i W_{λ_i}(t) 也是非负上鞅。
- Ville 不等式：P_H0(∃t, E_t ≥ 1/α) ≤ α，支持任意可选停止。
- 任意时刻 p 值：p_t = min(1, 1 / max_{s≤t} E_s)。

本模块全程在 log 域累计，财富归零（因子恰为 0）后该轨道永久为 0；
支持逐点/批量追加与 JSON 序列化，恢复时 m 与 λ 网格不可更改。

此外提供双侧、候选均值网格版本 :class:`TwoSidedMeanEvidence`：
对调用者给定的有限候选均值网格 {μ_i}，每个候选同时维护上尾财富
W⁺=∏(1+λ(x−μ_i)) 与下尾财富 W⁻=∏(1−λ(x−μ_i))，双侧 α 固定平分
（每侧阈值 2/α），并返回尚未被任一侧拒绝的候选集合。

只依赖 Python 3.14 标准库。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional, Sequence

__all__ = [
    "CandidateEvidence",
    "EvidenceSnapshot",
    "MeanEvidenceProcess",
    "TwoSidedMeanEvidence",
    "TwoSidedSnapshot",
    "STATE_VERSION",
]

STATE_VERSION = 1
_UNSET = object()


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------

def _finite_real(name: str, value) -> float:
    """返回 float(value)；拒绝 bool、非数值类型与 NaN/Infinity。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{name}: 需要 int/float 有限实数，实际得到 {type(value).__name__}"
        )
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(f"{name}: 拒绝 NaN/Infinity")
    return f


def _required_finite(data: dict, key: str) -> float:
    """从状态字典取必须存在的有限实数字段；缺失按 ValueError 处理。"""
    if key not in data:
        raise ValueError(f"state: 缺少字段 {key!r}")
    return _finite_real(f"state.{key}", data[key])


def _validate_config(m, lambdas, alpha) -> tuple[float, tuple[float, ...], float]:
    m_f = _finite_real("m", m)
    if not 0.0 < m_f < 1.0:
        raise ValueError(f"m: 必须严格位于 (0,1)，得到 {m_f!r}")

    if isinstance(lambdas, (str, bytes)) or not isinstance(lambdas, Sequence):
        raise TypeError(
            "lambdas: 需要有限序列（list/tuple），实际得到 "
            f"{type(lambdas).__name__}"
        )
    if len(lambdas) == 0:
        raise ValueError("lambdas: 网格不能为空")
    lam_max = 1.0 / m_f
    grid: list[float] = []
    for i, lam in enumerate(lambdas):
        f = _finite_real(f"lambdas[{i}]", lam)
        if f < 0.0 or f > lam_max:
            raise ValueError(
                f"lambdas[{i}]: 必须满足 0 <= lambda <= 1/m = {lam_max!r}，"
                f"得到 {f!r}"
            )
        grid.append(f)

    a_f = _finite_real("alpha", alpha)
    if not 0.0 < a_f < 1.0:
        raise ValueError(f"alpha: 必须严格位于 (0,1)，得到 {a_f!r}")

    return m_f, tuple(grid), a_f


# ---------------------------------------------------------------------------
# 公开快照
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceSnapshot:
    """t 时刻证据状态的不可变快照。

    t                 : 已处理观测个数（t=0 时 E_0 = 1）
    evidence          : E_t（直接 exp(log_evidence)，极大时可能为 inf，
                        请优先使用 log_evidence）
    log_evidence      : log E_t；所有财富轨道均为 0 时为 -inf
    max_evidence      : max_{s≤t} E_s（t=0 时为 1）
    max_log_evidence  : log(max_evidence)
    p_value           : min(1, 1/max_evidence)
    threshold         : 1/alpha
    stopped           : 是否已越过 threshold
    stop_time         : 首次 E_t ≥ 1/alpha 的时刻；未越过为 None
    """

    t: int
    evidence: float
    log_evidence: float
    max_evidence: float
    max_log_evidence: float
    p_value: float
    threshold: float
    stopped: bool
    stop_time: Optional[int]


# ---------------------------------------------------------------------------
# 核心过程
# ---------------------------------------------------------------------------

class MeanEvidenceProcess:
    """固定 m 与 λ 网格的在线均值证据过程。

    参数
    ----
    m : float, 0 < m < 1
        H0 条件均值上界。
    lambdas : 非空有限序列，每项 0 ≤ λ_i ≤ 1/m
        均匀混合使用的固定 λ 网格；重复值按重复计数（不做静默去重）。
    alpha : float, 0 < alpha < 1
        可选停止阈值 1/alpha。

    构造后用 :meth:`update`（单点）或 :meth:`extend`（批量、原子）追加观测；
    :meth:`snapshot` 取当前快照；:meth:`to_dict`/:meth:`to_json` 序列化，
    :meth:`from_dict`/:meth:`from_json` 恢复，恢复时 m/网格/alpha 不可更改。
    """

    def __init__(self, m, lambdas: Sequence, alpha):
        m_f, grid, a_f = _validate_config(m, lambdas, alpha)
        self._m: float = m_f
        self._lambdas: tuple[float, ...] = grid
        self._alpha: float = a_f
        self._threshold: float = 1.0 / a_f
        self._log_threshold: float = math.log(self._threshold)
        self._t: int = 0
        # 每条轨道的 log 财富；初始 0（W=1），归零后为 -inf 且永久保持。
        self._log_w: list[float] = [0.0] * len(grid)
        self._log_e: float = 0.0          # log E_0 = log 1 = 0
        self._max_log_e: float = 0.0
        self._stop_time: Optional[int] = None

    # ----- 构造与恢复 ------------------------------------------------------

    @classmethod
    def start(cls, m, lambdas: Sequence, alpha) -> "MeanEvidenceProcess":
        """显式起始（与构造器等价，语义化名称）。"""
        return cls(m, lambdas, alpha)

    @classmethod
    def from_dict(
        cls,
        data: dict,
        *,
        m=_UNSET,
        lambdas=_UNSET,
        alpha=_UNSET,
    ) -> "MeanEvidenceProcess":
        """从 :meth:`to_dict` 的字典恢复。

        m/lambdas/alpha 为可选的“配置断言”：提供时必须与序列化状态中的值
        完全一致，否则抛出 ValueError——恢复后不允许变更 m 或 λ 网格。
        """
        if not isinstance(data, dict):
            raise TypeError(f"state: 需要 dict，实际得到 {type(data).__name__}")
        if data.get("version") != STATE_VERSION:
            raise ValueError(
                f"state.version: 仅支持 {STATE_VERSION}，得到 {data.get('version')!r}"
            )

        m_s = _required_finite(data, "m")
        if not 0.0 < m_s < 1.0:
            raise ValueError(f"state.m: 越界 {m_s!r}")

        raw_grid = data.get("lambdas")
        if not isinstance(raw_grid, list) or not raw_grid:
            raise ValueError("state.lambdas: 必须为非空 list")
        lam_max = 1.0 / m_s
        grid: list[float] = []
        for i, lam in enumerate(raw_grid):
            f = _finite_real(f"state.lambdas[{i}]", lam)
            if f < 0.0 or f > lam_max:
                raise ValueError(f"state.lambdas[{i}]: 越界 {f!r}")
            grid.append(f)

        a_s = _required_finite(data, "alpha")
        if not 0.0 < a_s < 1.0:
            raise ValueError(f"state.alpha: 越界 {a_s!r}")

        # 调用方断言：恢复时不得更换配置。
        if m is not _UNSET and _finite_real("m", m) != m_s:
            raise ValueError(
                f"m: 恢复断言不一致（状态中 m={m_s!r}，给定 {float(m)!r}）"
            )
        if lambdas is not _UNSET:
            want = _validate_config(m_s, lambdas, a_s)[1]
            if want != tuple(grid):
                raise ValueError(
                    "lambdas: 恢复断言不一致，恢复后不允许更改 λ 网格 "
                    f"（状态中 {grid}，给定 {list(want)}）"
                )
        if alpha is not _UNSET and _finite_real("alpha", alpha) != a_s:
            raise ValueError(
                f"alpha: 恢复断言不一致（状态中 alpha={a_s!r}，"
                f"给定 {float(alpha)!r}）"
            )

        t = data.get("t")
        if isinstance(t, bool) or not isinstance(t, int) or t < 0:
            raise ValueError(f"state.t: 需要非负 int，得到 {t!r}")

        raw_lw = data.get("log_wealth")
        if not isinstance(raw_lw, list) or len(raw_lw) != len(grid):
            raise ValueError("state.log_wealth: 长度必须与 λ 网格一致")
        log_w: list[float] = []
        for i, v in enumerate(raw_lw):
            if v is None:
                log_w.append(-math.inf)
                continue
            # 有限即可；财富允许增长，故 log 值可正，无上界约束。
            f = _finite_real(f"state.log_wealth[{i}]", v)
            log_w.append(f)

        max_log = _required_finite(data, "max_log_evidence")
        if max_log < 0.0:
            raise ValueError("state.max_log_evidence: 不得小于 0（E_0=1）")

        stop = data.get("stop_time")
        if stop is not None:
            if isinstance(stop, bool) or not isinstance(stop, int) or not 1 <= stop <= t:
                raise ValueError(f"state.stop_time: 非法 {stop!r}")

        obj = cls.__new__(cls)
        obj._m = m_s
        obj._lambdas = tuple(grid)
        obj._alpha = a_s
        obj._threshold = 1.0 / a_s
        obj._log_threshold = math.log(obj._threshold)
        obj._t = t
        obj._log_w = log_w
        obj._max_log_e = max_log
        obj._stop_time = stop
        obj._log_e = obj._compute_log_evidence()
        if obj._max_log_e + 0.0 < obj._log_e:
            raise ValueError("state: max_log_evidence 与 log_wealth 不一致")
        if t == 0 and obj._max_log_e != 0.0:
            # t=0 时 E_0 恒为 1，历史峰值必为 0，无任何歧义空间。
            raise ValueError("state: t=0 时 max_log_evidence 必须恰为 0")
        if stop is not None and obj._max_log_e < obj._log_threshold:
            raise ValueError("state: stop_time 与 max_log_evidence 不一致（未越界却已停止）")
        return obj

    @classmethod
    def from_json(cls, text: str, **assert_kwargs) -> "MeanEvidenceProcess":
        """:meth:`from_dict` 的 JSON 字符串版本。"""
        if not isinstance(text, (str, bytes)):
            raise TypeError("text: 需要 str")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"state: JSON 解析失败: {exc}") from exc
        return cls.from_dict(data, **assert_kwargs)

    # ----- 序列化 ----------------------------------------------------------

    def to_dict(self) -> dict:
        """返回可 json 序列化的普通 dict；归零轨道以 null 表示。"""
        return {
            "version": STATE_VERSION,
            "m": self._m,
            "lambdas": list(self._lambdas),
            "alpha": self._alpha,
            "t": self._t,
            "log_wealth": [None if lw == -math.inf else lw for lw in self._log_w],
            "max_log_evidence": self._max_log_e,
            "stop_time": self._stop_time,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    # ----- 属性与快照 ------------------------------------------------------

    @property
    def m(self) -> float:
        return self._m

    @property
    def lambdas(self) -> tuple[float, ...]:
        return self._lambdas

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def t(self) -> int:
        return self._t

    def log_wealth(self) -> tuple[float, ...]:
        """各 λ 轨道当前 log 财富（归零轨道为 -inf）。"""
        return tuple(self._log_w)

    def _compute_log_evidence(self) -> float:
        """log E_t = hi + log( (1/K) Σ exp(logW_i - hi) )，跳过归零轨道。"""
        k = len(self._log_w)
        alive = [lw for lw in self._log_w if lw != -math.inf]
        if not alive:
            return -math.inf
        hi = max(alive)
        # fsum 降低求和误差；exp 之差 ≤ 0 不溢出。
        s = math.fsum(math.exp(lw - hi) for lw in alive)
        return hi + math.log(s / k)

    def snapshot(self) -> EvidenceSnapshot:
        log_e = self._log_e
        max_log = self._max_log_e
        if log_e == -math.inf:
            evidence = 0.0
        else:
            try:
                evidence = math.exp(log_e)
            except OverflowError:
                evidence = math.inf  # 超过 double 上界；精确量级请看 log_evidence
        try:
            max_evidence = math.exp(max_log)
        except OverflowError:
            max_evidence = math.inf
        # exp(-max_log) 下溢到 0.0 即 p=0；max_log=0 时 exp(0)=1 被 min 截为 1。
        p_value = min(1.0, math.exp(-max_log))
        return EvidenceSnapshot(
            t=self._t,
            evidence=evidence,
            log_evidence=log_e,
            max_evidence=max_evidence,
            max_log_evidence=max_log,
            p_value=p_value,
            threshold=self._threshold,
            stopped=self._stop_time is not None,
            stop_time=self._stop_time,
        )

    # ----- 追加观测 --------------------------------------------------------

    def _validate_x(self, x) -> float:
        f = _finite_real("x", x)
        if not 0.0 <= f <= 1.0:
            raise ValueError(f"x: 必须位于 [0,1]，得到 {f!r}")
        return f

    def _advance_one(self, x: float) -> None:
        """前置：x 已校验。推进一个观测（内部，无校验）。"""
        dx = x - self._m
        for i, lam in enumerate(self._lambdas):
            lw = self._log_w[i]
            if lw == -math.inf:
                # 归零轨道：0 乘任何非负因子仍为 0，永久跳过。
                continue
            # 单次 fma + 一次舍入；极端 λ 处舍入可能得到微小负值，
            # 真实因子非负，按 0 处理（财富归零，见 README 误差口径）。
            factor = math.fma(lam, dx, 1.0)
            if factor <= 0.0:
                self._log_w[i] = -math.inf
            else:
                self._log_w[i] = lw + math.log(factor)
        self._t += 1
        self._log_e = self._compute_log_evidence()
        if self._log_e > self._max_log_e:
            self._max_log_e = self._log_e
        if self._stop_time is None and self._log_e >= self._log_threshold:
            self._stop_time = self._t

    def update(self, x) -> EvidenceSnapshot:
        """追加一个合法观测 x ∈ [0,1]，返回推进后的快照。"""
        x_f = self._validate_x(x)
        self._advance_one(x_f)
        return self.snapshot()

    def extend(self, xs) -> EvidenceSnapshot:
        """批量追加。

        原子性：先完整校验整个批次（含迭代器，会先物化为 list），
        任一元素非法则抛错且本过程状态保持不变。
        """
        if isinstance(xs, (str, bytes)):
            raise TypeError(f"xs: 需要可迭代的实数序列，实际得到 {type(xs).__name__}")
        validated: list[float] = []
        try:
            iterator = iter(xs)
        except TypeError as exc:
            raise TypeError(f"xs: 不可迭代: {type(xs).__name__}") from exc
        for i, x in enumerate(iterator):
            f = _finite_real(f"xs[{i}]", x)
            if not 0.0 <= f <= 1.0:
                raise ValueError(f"xs[{i}]: 必须位于 [0,1]，得到 {f!r}")
            validated.append(f)
        for x in validated:
            self._advance_one(x)
        return self.snapshot()


# ---------------------------------------------------------------------------
# 双侧候选均值网格
# ---------------------------------------------------------------------------

TWO_SIDED_KIND = "two_sided_mean_evidence"
TWO_SIDED_STATE_VERSION = 1


def _validate_mu_grid(mus) -> tuple[float, ...]:
    """候选均值网格：非空有限序列，每项严格 0 < μ < 1；重复值按重复计数。"""
    if isinstance(mus, (str, bytes)) or not isinstance(mus, Sequence):
        raise TypeError(
            "mus: 需要有限序列（list/tuple），实际得到 "
            f"{type(mus).__name__}"
        )
    if len(mus) == 0:
        raise ValueError("mus: 候选均值网格不能为空")
    grid: list[float] = []
    for i, mu in enumerate(mus):
        f = _finite_real(f"mus[{i}]", mu)
        if not 0.0 < f < 1.0:
            raise ValueError(f"mus[{i}]: 必须严格位于 (0,1)，得到 {f!r}")
        grid.append(f)
    return tuple(grid)


@dataclass(frozen=True)
class CandidateEvidence:
    """单个候选均值 μ 在 t 时刻的双侧证据快照。

    上尾检验 H0⁺：E[x_t|F_{t-1}] ≤ μ，财富只在 x 偏大时增长；
    下尾检验 H0⁻：E[x_t|F_{t-1}] ≥ μ，财富只在 x 偏小时增长。
    每侧使用水平 α/2（阈值 2/α）；两侧均为非负上鞅，联合界控制双侧水平 α。

    upper_evidence/log/p       : 上尾混合 E⁺_t、log E⁺_t、原始单尾 p
                                 （min(1, 1/max E⁺)，阈值口径为 α/2）
    lower_*                    : 下尾对应量
    max_upper_log_evidence 等  : 各侧运行期峰值 log
    p_value_two_sided          : Bonferroni 合并 p = min(1, 2·min(p⁺, p⁻))，
                                 恰在候选被任一侧拒绝时 ≤ α
    rejected/reject_time/reject_side : 任一侧首次越过 2/α；
                                 reject_side 为 "upper"/"lower"；
                                 同一时刻两侧同时越过时记 "upper"
    """

    mu: float
    upper_evidence: float
    upper_log_evidence: float
    upper_max_evidence: float
    max_upper_log_evidence: float
    upper_p_value: float
    lower_evidence: float
    lower_log_evidence: float
    lower_max_evidence: float
    max_lower_log_evidence: float
    lower_p_value: float
    p_value_two_sided: float
    rejected: bool
    reject_time: Optional[int]
    reject_side: Optional[str]


@dataclass(frozen=True)
class TwoSidedSnapshot:
    """双侧候选网格在 t 时刻的整体快照。

    candidates : 按构造顺序排列的每个候选的 :class:`CandidateEvidence`
    surviving  : 尚未被任一侧拒绝的候选 μ（保持构造顺序，含重复值）
    rejected   : 已被拒绝的候选 μ
    threshold  : 每侧阈值 2/α
    alpha_per_side : α/2
    """

    t: int
    candidates: tuple[CandidateEvidence, ...]
    surviving: tuple[float, ...]
    rejected: tuple[float, ...]
    threshold: float
    alpha_per_side: float


class TwoSidedMeanEvidence:
    """有限候选均值网格上的双侧（上下尾）可下注序贯检验。

    参数
    ----
    mus : 非空有限序列，每项严格 0 < μ_i < 1
        调用者给定的候选均值网格；结论只对网格上的候选成立，
        网格之外的均值无法评估（见 README“网格限制”）。
    lambdas : 非空有限序列，每项有限且
        0 ≤ λ ≤ min_i min(1/μ_i, 1/(1−μ_i))。
        上下尾共用同一固定 λ 网格：上尾因子 1+λ(x−μ_i) 非负要求 λ ≤ 1/μ_i，
        下尾因子 1−λ(x−μ_i) 非负要求 λ ≤ 1/(1−μ_i)，故取所有候选的公共上界。
    alpha : 0 < α < 1
        双侧总水平，固定平分为每侧 α/2；每侧 Ville 阈值为 2/α，
        联合界给出双侧第一类错误 ≤ α。平分在构造时固定，不做自适应分配。

    下注仅依赖过去：t 时刻的财富是 x_1..x_t 的函数，第 t 步乘数只用到
    当前观测与既有财富，不使用任何未来信息（允许观测自适应、可非同分布）。

    内部每个候选复用两个 :class:`MeanEvidenceProcess`：上尾以 m=μ_i 直接
    吃 x；下尾以 m=1−μ_i 吃 y=1−x（因子恒等
    1+λ((1−x)−(1−μ_i)) ≡ 1−λ(x−μ_i)）。
    """

    def __init__(self, mus, lambdas: Sequence, alpha):
        mu_grid = _validate_mu_grid(mus)
        a_f = _finite_real("alpha", alpha)
        if not 0.0 < a_f < 1.0:
            raise ValueError(f"alpha: 必须严格位于 (0,1)，得到 {a_f!r}")
        # 公共 λ 上界：所有候选的上尾（1/μ）与下尾（1/(1−μ)）都要满足。
        lam_cap = min(
            min(1.0 / mu, 1.0 / (1.0 - mu)) for mu in mu_grid
        )
        if isinstance(lambdas, (str, bytes)) or not isinstance(lambdas, Sequence):
            raise TypeError(
                "lambdas: 需要有限序列（list/tuple），实际得到 "
                f"{type(lambdas).__name__}"
            )
        if len(lambdas) == 0:
            raise ValueError("lambdas: 网格不能为空")
        grid: list[float] = []
        for i, lam in enumerate(lambdas):
            f = _finite_real(f"lambdas[{i}]", lam)
            if f < 0.0 or f > lam_cap:
                raise ValueError(
                    f"lambdas[{i}]: 必须满足 0 <= lambda <= {lam_cap!r}"
                    "（所有候选上下尾因子非负的公共上界 "
                    "min_i min(1/mu_i, 1/(1-mu_i))），"
                    f"得到 {f!r}"
                )
            grid.append(f)

        self._mus: tuple[float, ...] = mu_grid
        self._lambdas: tuple[float, ...] = tuple(grid)
        self._alpha: float = a_f
        self._alpha_half: float = a_f / 2.0
        self._threshold: float = 2.0 / a_f
        half = self._alpha_half
        self._upper: list[MeanEvidenceProcess] = [
            MeanEvidenceProcess(mu, grid, half) for mu in mu_grid
        ]
        # 下尾：镜像成 m=1−μ、观测 1−x 的上尾过程。
        self._lower: list[MeanEvidenceProcess] = [
            MeanEvidenceProcess(1.0 - mu, grid, half) for mu in mu_grid
        ]

    # ----- 构造与恢复 ------------------------------------------------------

    @classmethod
    def start(cls, mus, lambdas: Sequence, alpha) -> "TwoSidedMeanEvidence":
        """显式起始（与构造器等价，语义化名称）。"""
        return cls(mus, lambdas, alpha)

    @classmethod
    def from_dict(cls, data: dict) -> "TwoSidedMeanEvidence":
        """从 :meth:`to_dict` 的字典恢复；mus/lambdas/alpha 不可更改。

        上下尾子状态交给 :meth:`MeanEvidenceProcess.from_dict` 做完整校验，
        本方法额外校验种类标识、数组长度、μ 与镜像 m、α/2、t 的一致性。
        """
        if not isinstance(data, dict):
            raise TypeError(f"state: 需要 dict，实际得到 {type(data).__name__}")
        if data.get("kind") != TWO_SIDED_KIND:
            raise ValueError(f"state.kind: 期望 {TWO_SIDED_KIND!r}，"
                             f"得到 {data.get('kind')!r}")
        if data.get("version") != TWO_SIDED_STATE_VERSION:
            raise ValueError(
                f"state.version: 仅支持 {TWO_SIDED_STATE_VERSION}，"
                f"得到 {data.get('version')!r}"
            )

        raw_mus = data.get("mus")
        if not isinstance(raw_mus, list) or not raw_mus:
            raise ValueError("state.mus: 必须为非空 list")
        mus = _validate_mu_grid(raw_mus)
        a_f = _required_finite(data, "alpha")
        if not 0.0 < a_f < 1.0:
            raise ValueError(f"state.alpha: 越界 {a_f!r}")
        raw_grid = data.get("lambdas")
        if not isinstance(raw_grid, list) or not raw_grid:
            raise ValueError("state.lambdas: 必须为非空 list")
        lam_cap = min(min(1.0 / mu, 1.0 / (1.0 - mu)) for mu in mus)
        grid: list[float] = []
        for i, lam in enumerate(raw_grid):
            f = _finite_real(f"state.lambdas[{i}]", lam)
            if f < 0.0 or f > lam_cap:
                raise ValueError(f"state.lambdas[{i}]: 越界 {f!r}")
            grid.append(f)
        grid_t = tuple(grid)

        t = data.get("t")
        if isinstance(t, bool) or not isinstance(t, int) or t < 0:
            raise ValueError(f"state.t: 需要非负 int，得到 {t!r}")

        upper_raw, lower_raw = data.get("upper"), data.get("lower")
        if not isinstance(upper_raw, list) or not isinstance(lower_raw, list):
            raise ValueError("state.upper/state.lower: 必须为 list")
        if len(upper_raw) != len(mus) or len(lower_raw) != len(mus):
            raise ValueError("state: 上下尾状态数必须与候选网格长度一致")

        obj = cls.__new__(cls)
        obj._mus = mus
        obj._lambdas = grid_t
        obj._alpha = a_f
        obj._alpha_half = a_f / 2.0
        obj._threshold = 2.0 / a_f
        upper: list[MeanEvidenceProcess] = []
        lower: list[MeanEvidenceProcess] = []
        for i, mu in enumerate(mus):
            up = MeanEvidenceProcess.from_dict(upper_raw[i])
            lo = MeanEvidenceProcess.from_dict(lower_raw[i])
            # 交叉校验：子过程配置必须与候选网格、平分 α 完全一致。
            if up.m != mu or lo.m != 1.0 - mu:
                raise ValueError(f"state: 候选 {i} 的子过程 m 与 μ 不一致")
            if up.lambdas != grid_t or lo.lambdas != grid_t:
                raise ValueError(f"state: 候选 {i} 的 λ 网格不一致")
            if up.alpha != obj._alpha_half or lo.alpha != obj._alpha_half:
                raise ValueError(f"state: 候选 {i} 的子过程 alpha 必须恰为 alpha/2")
            if up.t != t or lo.t != t:
                raise ValueError(f"state: 候选 {i} 的 t 不一致")
            upper.append(up)
            lower.append(lo)
        obj._upper = upper
        obj._lower = lower
        return obj

    @classmethod
    def from_json(cls, text: str) -> "TwoSidedMeanEvidence":
        """:meth:`from_dict` 的 JSON 字符串版本。"""
        if not isinstance(text, (str, bytes)):
            raise TypeError("text: 需要 str")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"state: JSON 解析失败: {exc}") from exc
        return cls.from_dict(data)

    # ----- 序列化 ----------------------------------------------------------

    def to_dict(self) -> dict:
        """返回可 json 序列化的普通 dict（上下尾各为 version=1 子状态）。"""
        return {
            "kind": TWO_SIDED_KIND,
            "version": TWO_SIDED_STATE_VERSION,
            "mus": list(self._mus),
            "lambdas": list(self._lambdas),
            "alpha": self._alpha,
            "t": self._upper[0].t,
            "upper": [p.to_dict() for p in self._upper],
            "lower": [p.to_dict() for p in self._lower],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    # ----- 属性与快照 ------------------------------------------------------

    @property
    def mus(self) -> tuple[float, ...]:
        return self._mus

    @property
    def lambdas(self) -> tuple[float, ...]:
        return self._lambdas

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def alpha_per_side(self) -> float:
        return self._alpha_half

    @property
    def threshold(self) -> float:
        """每侧 Ville 阈值 2/α。"""
        return self._threshold

    @property
    def t(self) -> int:
        return self._upper[0].t

    def candidate_log_wealth(self, i: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """第 i 个候选的 (上尾各 λ log 财富, 下尾各 λ log 财富)。"""
        if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(self._mus):
            raise IndexError(f"i: 候选下标越界，得到 {i!r}")
        return self._upper[i].log_wealth(), self._lower[i].log_wealth()

    def _candidate_evidence(self, i: int) -> CandidateEvidence:
        up = self._upper[i].snapshot()
        lo = self._lower[i].snapshot()
        # 任一侧首次越过即拒绝；同一时刻两侧都越过记上尾。
        if up.stop_time is None and lo.stop_time is None:
            reject_time: Optional[int] = None
            reject_side: Optional[str] = None
        elif up.stop_time is None:
            reject_time, reject_side = lo.stop_time, "lower"
        elif lo.stop_time is None:
            reject_time, reject_side = up.stop_time, "upper"
        elif up.stop_time <= lo.stop_time:
            reject_time, reject_side = up.stop_time, "upper"
        else:
            reject_time, reject_side = lo.stop_time, "lower"
        p_two = min(1.0, 2.0 * min(up.p_value, lo.p_value))
        return CandidateEvidence(
            mu=self._mus[i],
            upper_evidence=up.evidence,
            upper_log_evidence=up.log_evidence,
            upper_max_evidence=up.max_evidence,
            max_upper_log_evidence=up.max_log_evidence,
            upper_p_value=up.p_value,
            lower_evidence=lo.evidence,
            lower_log_evidence=lo.log_evidence,
            lower_max_evidence=lo.max_evidence,
            max_lower_log_evidence=lo.max_log_evidence,
            lower_p_value=lo.p_value,
            p_value_two_sided=p_two,
            rejected=reject_time is not None,
            reject_time=reject_time,
            reject_side=reject_side,
        )

    def snapshot(self) -> TwoSidedSnapshot:
        cands = tuple(self._candidate_evidence(i) for i in range(len(self._mus)))
        surviving = tuple(c.mu for c in cands if not c.rejected)
        rejected = tuple(c.mu for c in cands if c.rejected)
        return TwoSidedSnapshot(
            t=self.t,
            candidates=cands,
            surviving=surviving,
            rejected=rejected,
            threshold=self._threshold,
            alpha_per_side=self._alpha_half,
        )

    # ----- 追加观测 --------------------------------------------------------

    def _advance_one(self, x: float) -> None:
        """前置：x 已校验。上下尾所有候选同步推进一步。"""
        for up, lo in zip(self._upper, self._lower):
            up._advance_one(x)
            lo._advance_one(1.0 - x)

    def update(self, x) -> TwoSidedSnapshot:
        """追加一个合法观测 x ∈ [0,1]，返回推进后的双侧快照。"""
        x_f = _finite_real("x", x)
        if not 0.0 <= x_f <= 1.0:
            raise ValueError(f"x: 必须位于 [0,1]，得到 {x_f!r}")
        self._advance_one(x_f)
        return self.snapshot()

    def extend(self, xs) -> TwoSidedSnapshot:
        """批量追加（原子）：先完整校验整批，任一非法则状态保持不变。"""
        if isinstance(xs, (str, bytes)):
            raise TypeError(f"xs: 需要可迭代的实数序列，实际得到 {type(xs).__name__}")
        try:
            iterator = iter(xs)
        except TypeError as exc:
            raise TypeError(f"xs: 不可迭代: {type(xs).__name__}") from exc
        validated: list[float] = []
        for i, x in enumerate(iterator):
            f = _finite_real(f"xs[{i}]", x)
            if not 0.0 <= f <= 1.0:
                raise ValueError(f"xs[{i}]: 必须位于 [0,1]，得到 {f!r}")
            validated.append(f)
        for x in validated:
            self._advance_one(x)
        return self.snapshot()
