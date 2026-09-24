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

只依赖 Python 3.14 标准库。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional, Sequence

__all__ = [
    "ConfidenceSnapshot",
    "EvidenceSnapshot",
    "MeanEvidenceProcess",
    "MeanGridConfidenceSet",
    "GRID_STATE_VERSION",
    "STATE_VERSION",
]

STATE_VERSION = 1
GRID_STATE_VERSION = 1
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
# 有限均值网格上的双侧置信集
# ---------------------------------------------------------------------------

def _validate_candidates(candidates) -> tuple[float, ...]:
    """候选均值网格：非空有限序列，每项严格位于 (0,1)，不允许重复。"""
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise TypeError(
            "candidates: 需要有限序列（list/tuple），实际得到 "
            f"{type(candidates).__name__}"
        )
    if len(candidates) == 0:
        raise ValueError("candidates: 候选均值网格不能为空")
    out: list[float] = []
    for i, c in enumerate(candidates):
        f = _finite_real(f"candidates[{i}]", c)
        if not 0.0 < f < 1.0:
            raise ValueError(f"candidates[{i}]: 必须严格位于 (0,1)，得到 {f!r}")
        if f in out:
            raise ValueError(f"candidates[{i}]: 候选均值重复 {f!r}")
        out.append(f)
    return tuple(out)


def _two_sided_lam_max(candidates: tuple[float, ...]) -> float:
    """同一 λ 网格要同时满足上尾（λ ≤ 1/m_j）与下尾（λ ≤ 1/(1-m_j)）。"""
    return min(min(1.0 / c, 1.0 / (1.0 - c)) for c in candidates)


def _validate_shared_lambdas(lambdas, lam_max: float) -> tuple[float, ...]:
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
        if f < 0.0 or f > lam_max:
            raise ValueError(
                f"lambdas[{i}]: 必须满足 0 <= lambda <= {lam_max!r} "
                f"（= min_j min(1/m_j, 1/(1-m_j))），得到 {f!r}"
            )
        grid.append(f)
    return tuple(grid)


@dataclass(frozen=True)
class ConfidenceSnapshot:
    """t 时刻有限网格双侧置信集状态的不可变快照。

    t                  : 已处理观测个数（t=0 时尚无任何拒绝）
    alpha              : 双侧总水平
    alpha_side         : 每侧水平，固定为 alpha/2（双侧平分）
    threshold          : 每侧拒绝阈值 2/alpha
    candidates         : 完整候选均值网格（构造时固定）
    rejected           : 已被拒绝的候选（按网格顺序）
    confidence_set     : 尚未被拒绝的候选（按网格顺序；可以为空——
                         空集表示真实均值不在给定网格内，见 README 网格限制）
    upper_log_evidence : 各候选上尾（H0: μ ≤ m_j）log 证据，与 candidates 对齐；
                         财富归零为 -inf
    lower_log_evidence : 各候选下尾（H0: μ ≥ m_j）log 证据，与 candidates 对齐
    rejection_times    : 各候选首次被拒绝的时刻（未拒绝为 None），与 candidates 对齐
    """

    t: int
    alpha: float
    alpha_side: float
    threshold: float
    candidates: tuple[float, ...]
    rejected: tuple[float, ...]
    confidence_set: tuple[float, ...]
    upper_log_evidence: tuple[float, ...]
    lower_log_evidence: tuple[float, ...]
    rejection_times: tuple[Optional[int], ...]


class MeanGridConfidenceSet:
    """有限候选均值网格上的双侧序贯置信集。

    参数
    ----
    candidates : 非空有限序列，每项严格位于 (0,1) 且不重复
        调用者给定的候选均值网格；置信集永远是该网格的子集（网格限制：
        网格外的均值不在本对象的表达范围内）。
    lambdas : 非空有限序列，每项 0 ≤ λ_i ≤ min_j min(1/m_j, 1/(1-m_j))
        两侧共固定的 λ 网格；重复值按重复计数。
        上界保证上尾因子 1+λ(x-m_j) 与下尾因子 1+λ(m_j-x) 对所有候选非负。
    alpha : float, 0 < alpha < 1
        双侧总水平；固定平分为每侧 alpha/2，每侧拒绝阈值 2/alpha。

    对每个候选 m_j 同时维护两条财富过程：

    - 上尾：检验 H0+：E[x_t|F_{t-1}] ≤ m_j，因子 1 + λ(x_t − m_j)；
    - 下尾：检验 H0-：E[x_t|F_{t-1}] ≥ m_j，因子 1 + λ(m_j − x_t)。

    下注策略是构造时固定的常数 λ——常数不依赖任何观测，因而自动满足
    “下注仅依赖过去”（关于 F_{t-1} 可预测）的要求，两条财富过程在各自
    零假设下都是非负上鞅。候选 m_j 在任一侧财富首次 ≥ 2/alpha 时被拒绝，
    拒绝不可逆（记录首次越界时刻）。由 Ville 不等式与并集界：若真实均值
    μ 在网格中，则 P(μ 曾被拒绝) ≤ alpha/2 + alpha/2 = alpha。

    用法与 :class:`MeanEvidenceProcess` 平行：:meth:`update`（单点）、
    :meth:`extend`（批量、原子）、:meth:`snapshot`、:meth:`confidence_set`、
    :meth:`to_dict`/:meth:`to_json` 序列化，:meth:`from_dict`/:meth:`from_json`
    恢复（恢复时 candidates/网格/alpha 不可更改）。
    """

    def __init__(self, candidates, lambdas: Sequence, alpha):
        cands = _validate_candidates(candidates)
        grid = _validate_shared_lambdas(lambdas, _two_sided_lam_max(cands))
        a_f = _finite_real("alpha", alpha)
        if not 0.0 < a_f < 1.0:
            raise ValueError(f"alpha: 必须严格位于 (0,1)，得到 {a_f!r}")
        self._candidates: tuple[float, ...] = cands
        self._lambdas: tuple[float, ...] = grid
        self._alpha: float = a_f
        self._alpha_side: float = a_f / 2.0
        self._threshold: float = 1.0 / self._alpha_side
        # 上尾直接检验 μ ≤ m_j；下尾等价于对 y_t = 1 - x_t 检验
        # E[y_t|F_{t-1}] ≤ 1 - m_j，复用同一上鞅实现。
        self._upper = [
            MeanEvidenceProcess(c, grid, self._alpha_side) for c in cands
        ]
        self._lower = [
            MeanEvidenceProcess(1.0 - c, grid, self._alpha_side) for c in cands
        ]

    # ----- 构造与恢复 ------------------------------------------------------

    @classmethod
    def start(cls, candidates, lambdas: Sequence, alpha) -> "MeanGridConfidenceSet":
        """显式起始（与构造器等价，语义化名称）。"""
        return cls(candidates, lambdas, alpha)

    @classmethod
    def from_dict(
        cls,
        data: dict,
        *,
        candidates=_UNSET,
        lambdas=_UNSET,
        alpha=_UNSET,
    ) -> "MeanGridConfidenceSet":
        """从 :meth:`to_dict` 的字典恢复。

        candidates/lambdas/alpha 为可选的“配置断言”：提供时必须与序列化
        状态中的值完全一致，否则抛出 ValueError——恢复后不允许变更候选
        网格、λ 网格或 alpha。
        """
        if not isinstance(data, dict):
            raise TypeError(f"state: 需要 dict，实际得到 {type(data).__name__}")
        if data.get("version") != GRID_STATE_VERSION:
            raise ValueError(
                f"state.version: 仅支持 {GRID_STATE_VERSION}，"
                f"得到 {data.get('version')!r}"
            )

        raw_cands = data.get("candidates")
        if not isinstance(raw_cands, list):
            raise ValueError("state.candidates: 必须为非空 list")
        cands = _validate_candidates(raw_cands)

        raw_grid = data.get("lambdas")
        if not isinstance(raw_grid, list):
            raise ValueError("state.lambdas: 必须为非空 list")
        grid = _validate_shared_lambdas(raw_grid, _two_sided_lam_max(cands))

        a_s = _required_finite(data, "alpha")
        if not 0.0 < a_s < 1.0:
            raise ValueError(f"state.alpha: 越界 {a_s!r}")
        alpha_side = a_s / 2.0

        t = data.get("t")
        if isinstance(t, bool) or not isinstance(t, int) or t < 0:
            raise ValueError(f"state.t: 需要非负 int，得到 {t!r}")

        sides: list[list[MeanEvidenceProcess]] = []
        for key in ("upper", "lower"):
            raw_side = data.get(key)
            if not isinstance(raw_side, list) or len(raw_side) != len(cands):
                raise ValueError(
                    f"state.{key}: 必须为与 candidates 等长的 list"
                )
            side: list[MeanEvidenceProcess] = []
            for j, sub in enumerate(raw_side):
                try:
                    side.append(MeanEvidenceProcess.from_dict(sub))
                except ValueError as exc:
                    raise ValueError(f"state.{key}[{j}]: {exc}") from exc
            sides.append(side)
        upper, lower = sides

        # 交叉校验：子过程配置必须与顶层状态一致，不能盲信序列化内容。
        for j, cand in enumerate(cands):
            up, low = upper[j], lower[j]
            if up.m != cand:
                raise ValueError(
                    f"state.upper[{j}].m={up.m!r} 与 candidates[{j}]={cand!r} 不一致"
                )
            if low.m != 1.0 - cand:
                raise ValueError(
                    f"state.lower[{j}].m={low.m!r} 与 1-candidates[{j}] 不一致"
                )
            if up.lambdas != grid or low.lambdas != grid:
                raise ValueError(f"state: 第 {j} 个子过程 λ 网格与顶层不一致")
            if up.alpha != alpha_side or low.alpha != alpha_side:
                raise ValueError(f"state: 第 {j} 个子过程 alpha 与 alpha/2 不一致")
            if up.t != t or low.t != t:
                raise ValueError(f"state: 第 {j} 个子过程 t 与顶层不一致")

        # 调用方断言：恢复时不得更换配置。
        if candidates is not _UNSET and _validate_candidates(candidates) != cands:
            raise ValueError("candidates: 恢复断言不一致，恢复后不允许更改候选网格")
        if lambdas is not _UNSET:
            want = _validate_shared_lambdas(lambdas, _two_sided_lam_max(cands))
            if want != grid:
                raise ValueError("lambdas: 恢复断言不一致，恢复后不允许更改 λ 网格")
        if alpha is not _UNSET and _finite_real("alpha", alpha) != a_s:
            raise ValueError("alpha: 恢复断言不一致，恢复后不允许更改 alpha")

        obj = cls.__new__(cls)
        obj._candidates = cands
        obj._lambdas = grid
        obj._alpha = a_s
        obj._alpha_side = alpha_side
        obj._threshold = 1.0 / alpha_side
        obj._upper = upper
        obj._lower = lower
        return obj

    @classmethod
    def from_json(cls, text: str, **assert_kwargs) -> "MeanGridConfidenceSet":
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
        """返回可 json 序列化的普通 dict（version=1 网格状态格式）。"""
        return {
            "version": GRID_STATE_VERSION,
            "candidates": list(self._candidates),
            "lambdas": list(self._lambdas),
            "alpha": self._alpha,
            "t": self.t,
            "upper": [p.to_dict() for p in self._upper],
            "lower": [p.to_dict() for p in self._lower],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    # ----- 属性与快照 ------------------------------------------------------

    @property
    def candidates(self) -> tuple[float, ...]:
        return self._candidates

    @property
    def lambdas(self) -> tuple[float, ...]:
        return self._lambdas

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def alpha_side(self) -> float:
        """每侧分得的水平，固定为 alpha/2。"""
        return self._alpha_side

    @property
    def t(self) -> int:
        return self._upper[0].t

    def snapshot(self) -> ConfidenceSnapshot:
        rejected: list[float] = []
        kept: list[float] = []
        times: list[Optional[int]] = []
        upper_logs: list[float] = []
        lower_logs: list[float] = []
        for j, cand in enumerate(self._candidates):
            up = self._upper[j].snapshot()
            low = self._lower[j].snapshot()
            upper_logs.append(up.log_evidence)
            lower_logs.append(low.log_evidence)
            stops = [s for s in (up.stop_time, low.stop_time) if s is not None]
            if stops:
                rejected.append(cand)
                times.append(min(stops))
            else:
                kept.append(cand)
                times.append(None)
        return ConfidenceSnapshot(
            t=self.t,
            alpha=self._alpha,
            alpha_side=self._alpha_side,
            threshold=self._threshold,
            candidates=self._candidates,
            rejected=tuple(rejected),
            confidence_set=tuple(kept),
            upper_log_evidence=tuple(upper_logs),
            lower_log_evidence=tuple(lower_logs),
            rejection_times=tuple(times),
        )

    def confidence_set(self) -> tuple[float, ...]:
        """当前尚未被拒绝的候选均值（按网格顺序；可以为空）。"""
        return self.snapshot().confidence_set

    # ----- 追加观测 --------------------------------------------------------

    def _validate_x(self, x) -> float:
        f = _finite_real("x", x)
        if not 0.0 <= f <= 1.0:
            raise ValueError(f"x: 必须位于 [0,1]，得到 {f!r}")
        return f

    def _advance_one(self, x: float) -> None:
        """前置：x 已校验。推进一个观测（内部，无校验）。"""
        y = 1.0 - x
        for j in range(len(self._candidates)):
            self._upper[j]._advance_one(x)
            self._lower[j]._advance_one(y)

    def update(self, x) -> ConfidenceSnapshot:
        """追加一个合法观测 x ∈ [0,1]，返回推进后的快照。"""
        x_f = self._validate_x(x)
        self._advance_one(x_f)
        return self.snapshot()

    def extend(self, xs) -> ConfidenceSnapshot:
        """批量追加。

        原子性：先完整校验整个批次（含迭代器，会先物化为 list），
        任一元素非法则抛错且本对象状态保持不变。
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
