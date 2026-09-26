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

另提供固定非负权重的多起点混合入口：每个可能起点重新运行同一财富过程，
尚未开启的分量以初始财富 1 保留其先验质量；该入口只给条件均值零假设的证据，
不把贡献最大的起点解释为已认证变点。

只依赖 Python 3.14 标准库。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional, Sequence

__all__ = [
    "ChangePointEvidenceProcess",
    "EvidenceSnapshot",
    "MeanEvidenceProcess",
    "MixedEvidenceSnapshot",
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


@dataclass(frozen=True)
class MixedEvidenceSnapshot:
    """固定起点权重混合后的证据快照。

    component_evidence : 第 s 个可能起点重新开始后的 E_t；尚未开启时为 1。
    contributions      : 各起点对混合值的实际加项 w_s E_s；未来起点为 w_s。
    active_starts      : 当前已经开启的分量数。
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
    active_starts: int
    component_evidence: tuple[float, ...]
    contributions: tuple[float, ...]


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


class ChangePointEvidenceProcess:
    """按固定权重混合多个可能重启起点的均值证据过程。

    可能起点用零基下标 ``s=0,...,max_starts-1`` 表示：第 ``s`` 个分量从第
    ``s+1`` 个观测开始重新运行原有 :class:`MeanEvidenceProcess`。在它开启前，
    该分量的财富恒为初始值 1，因此当前混合值始终包含 ``w_s`` 这部分质量；
    不能在早期只按已开启权重重归一化。

    权重在构造时固定且总和必须为 1。越过阈值只记录首次报警时刻，之后仍继续
    更新所有已开启分量，并按原计划开启新分量，不依据数据重分配权重。
    """

    def __init__(self, m, lambdas: Sequence, alpha, weights: Sequence, max_starts: int):
        m_f, grid, a_f = _validate_config(m, lambdas, alpha)
        weights_tuple = _validate_weights(weights)
        if isinstance(max_starts, bool) or not isinstance(max_starts, int):
            raise TypeError(
                "max_starts: 需要正 int，实际得到 "
                f"{type(max_starts).__name__}"
            )
        if max_starts <= 0:
            raise ValueError(f"max_starts: 必须为正整数，得到 {max_starts!r}")
        if len(weights_tuple) != max_starts:
            raise ValueError(
                f"weights: 长度必须等于 max_starts={max_starts}，"
                f"得到 {len(weights_tuple)}"
            )

        self._m = m_f
        self._lambdas = grid
        self._alpha = a_f
        self._threshold = 1.0 / a_f
        self._log_threshold = math.log(self._threshold)
        self._weights = weights_tuple
        self._max_starts = max_starts
        self._components: list[MeanEvidenceProcess] = [
            MeanEvidenceProcess(m_f, grid, a_f)
        ]
        self._t = 0
        self._max_log_e = 0.0
        self._stop_time: Optional[int] = None

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
    def weights(self) -> tuple[float, ...]:
        return self._weights

    @property
    def max_starts(self) -> int:
        return self._max_starts

    @property
    def t(self) -> int:
        return self._t

    @property
    def active_starts(self) -> int:
        return len(self._components)

    def component_log_evidence(self) -> tuple[float, ...]:
        """返回所有起点当前 log E；尚未开启的起点为 log(1)=0。"""
        values = [0.0] * self._max_starts
        for i, proc in enumerate(self._components):
            values[i] = proc.snapshot().log_evidence
        return tuple(values)

    def component_evidence(self) -> tuple[float, ...]:
        """返回所有起点当前 E；尚未开启的起点为初始值 1。"""
        values = [1.0] * self._max_starts
        for i, proc in enumerate(self._components):
            values[i] = proc.snapshot().evidence
        return tuple(values)

    def contributions(self) -> tuple[float, ...]:
        """返回各起点对当前混合值的加项 ``w_s * E_s``。"""
        evidence = self.component_evidence()
        return tuple(
            0.0 if weight == 0.0 else weight * value
            for weight, value in zip(self._weights, evidence)
        )

    def _mixed_log_evidence(self) -> float:
        terms: list[float] = []
        for s, proc in enumerate(self._components):
            weight = self._weights[s]
            log_e = proc.snapshot().log_evidence
            if weight > 0.0 and log_e != -math.inf:
                terms.append(math.log(weight) + log_e)
        for s in range(len(self._components), self._max_starts):
            weight = self._weights[s]
            if weight > 0.0:
                terms.append(math.log(weight))
        if not terms:
            return -math.inf
        hi = max(terms)
        total = math.fsum(math.exp(value - hi) for value in terms)
        return hi + math.log(total)

    def snapshot(self) -> MixedEvidenceSnapshot:
        log_e = self._mixed_log_evidence()
        if log_e == -math.inf:
            evidence = 0.0
        else:
            try:
                evidence = math.exp(log_e)
            except OverflowError:
                evidence = math.inf
        try:
            max_evidence = math.exp(self._max_log_e)
        except OverflowError:
            max_evidence = math.inf
        return MixedEvidenceSnapshot(
            t=self._t,
            evidence=evidence,
            log_evidence=log_e,
            max_evidence=max_evidence,
            max_log_evidence=self._max_log_e,
            p_value=min(1.0, math.exp(-self._max_log_e)),
            threshold=self._threshold,
            stopped=self._stop_time is not None,
            stop_time=self._stop_time,
            active_starts=len(self._components),
            component_evidence=self.component_evidence(),
            contributions=self.contributions(),
        )

    def _validate_x(self, x) -> float:
        f = _finite_real("x", x)
        if not 0.0 <= f <= 1.0:
            raise ValueError(f"x: 必须位于 [0,1]，得到 {f!r}")
        return f

    def _advance_one(self, x: float) -> None:
        if self._t > 0 and len(self._components) < self._max_starts:
            self._components.append(
                MeanEvidenceProcess(self._m, self._lambdas, self._alpha)
            )
        for proc in self._components:
            proc.update(x)
        self._t += 1
        log_e = self._mixed_log_evidence()
        if log_e > self._max_log_e:
            self._max_log_e = log_e
        if self._stop_time is None and log_e >= self._log_threshold:
            self._stop_time = self._t

    def update(self, x) -> MixedEvidenceSnapshot:
        """追加一个观测；先更新旧分量，再开启以当前样本为起点的新分量。"""
        x_f = self._validate_x(x)
        self._advance_one(x_f)
        return self.snapshot()

    def extend(self, xs) -> MixedEvidenceSnapshot:
        """批量追加；先校验全部输入，失败时状态不变。"""
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


def _validate_weights(weights: Sequence) -> tuple[float, ...]:
    if isinstance(weights, (str, bytes)) or not isinstance(weights, Sequence):
        raise TypeError(
            "weights: 需要有限非负实数序列，实际得到 "
            f"{type(weights).__name__}"
        )
    if not weights:
        raise ValueError("weights: 不能为空")
    out: list[float] = []
    for i, weight in enumerate(weights):
        f = _finite_real(f"weights[{i}]", weight)
        if f < 0.0:
            raise ValueError(f"weights[{i}]: 必须非负，得到 {f!r}")
        out.append(f)
    total = math.fsum(out)
    if total != 1.0:
        raise ValueError(f"weights: 总和必须固定为 1，得到 {total!r}")
    return tuple(out)
