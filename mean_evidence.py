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
    "EvidenceSnapshot",
    "MeanEvidenceProcess",
    "FiniteBettingStrategy",
    "PredictableBettingProcess",
    "SealedBet",
    "STATE_VERSION",
    "PREDICTABLE_STATE_VERSION",
]

STATE_VERSION = 1
PREDICTABLE_STATE_VERSION = 1
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
# 可预测（只依赖过去）的自适应下注：有限状态表策略 + 封存式轮次接口
# ---------------------------------------------------------------------------


class FiniteBettingStrategy:
    """仓库内有限状态表定义的确定性下注策略。

    策略只由不可变的有限数据组成（无回调、无字符串代码、无外部 I/O）：

    - states        : 非空有限状态名（str）序列；
    - initial_state : 初始状态（第 1 轮下注前所处状态）；
    - bet_lambdas   : 每个状态封存一个下注值 {状态: lambda}，即处于该状态、
      看到本轮 x 之前必须封存的 lambda；
    - edges         : 状态转移表 {状态: {反馈档位: 下一状态}}，观察 x 后按其
      落入的显式半开区间档位选择下一状态；下一轮 lambda 由下一状态唯一决定，
      即“状态与上一轮离散反馈选择下一状态和 lambda”；
    - cutpoints     : 分箱切点 0 = c0 < c1 < ... < c_n = 1，档位 k 为半开
      区间 [c_k, c_{k+1})，最后一档为 [c_{n-1}, 1]；区间覆盖 [0,1] 且互不重叠。

    每轮下注顺序被强制为：先用已有历史（当前状态）查表生成并封存 lambda，
    再接收这一轮 x；x 只影响之后的轮次，不可能回看当前 x 挑收益最大的
    lambda。
    """

    def __init__(self, states, initial_state, bet_lambdas, edges, cutpoints):
        states_t = self._validate_states(states)
        if not isinstance(initial_state, str) or initial_state not in states_t:
            raise ValueError(
                "initial_state: 必须是 states 中的 str，"
                f"得到 {initial_state!r}"
            )
        cuts = self._validate_cutpoints(cutpoints)
        lams = self._validate_lambdas(bet_lambdas, states_t)
        table = self._validate_edges(edges, states_t, len(cuts) - 1)
        self._states: tuple[str, ...] = states_t
        self._initial_state: str = initial_state
        self._lambdas: dict[str, float] = lams
        self._edges: dict[str, dict[int, str]] = table
        self._cutpoints: tuple[float, ...] = cuts

    @staticmethod
    def _validate_states(states) -> tuple[str, ...]:
        if isinstance(states, (str, bytes)) or not isinstance(states, Sequence):
            raise TypeError(
                "states: 需要非空有限 str 序列，实际得到 "
                f"{type(states).__name__}"
            )
        if len(states) == 0:
            raise ValueError("states: 状态集合不能为空")
        seen: set[str] = set()
        out: list[str] = []
        for i, s in enumerate(states):
            if not isinstance(s, str):
                raise TypeError(
                    f"states[{i}]: 状态名必须为 str，得到 {type(s).__name__}"
                )
            if not s:
                raise ValueError(f"states[{i}]: 状态名不能为空串")
            if s in seen:
                raise ValueError(f"states[{i}]: 状态名重复 {s!r}")
            seen.add(s)
            out.append(s)
        return tuple(out)

    @staticmethod
    def _validate_cutpoints(cutpoints) -> tuple[float, ...]:
        if isinstance(cutpoints, (str, bytes)) or not isinstance(
            cutpoints, Sequence
        ):
            raise TypeError(
                "cutpoints: 需要有限实数序列，实际得到 "
                f"{type(cutpoints).__name__}"
            )
        if len(cutpoints) < 2:
            raise ValueError("cutpoints: 至少需要两个切点（区间端点）")
        cuts: list[float] = []
        for i, c in enumerate(cutpoints):
            cuts.append(_finite_real(f"cutpoints[{i}]", c))
        if cuts[0] != 0.0 or cuts[-1] != 1.0:
            raise ValueError("cutpoints: 必须以 0.0 开始、1.0 结束")
        for i in range(1, len(cuts)):
            if cuts[i] <= cuts[i - 1]:
                raise ValueError(
                    f"cutpoints[{i}]: 切点必须严格递增，"
                    f"{cuts[i]!r} <= {cuts[i - 1]!r}"
                )
        return tuple(cuts)

    @staticmethod
    def _validate_lambdas(bet_lambdas, states) -> dict[str, float]:
        if not isinstance(bet_lambdas, dict):
            raise TypeError(
                "bet_lambdas: 需要 dict {状态: lambda}，实际得到 "
                f"{type(bet_lambdas).__name__}"
            )
        unknown = set(bet_lambdas) - set(states)
        missing = set(states) - set(bet_lambdas)
        if unknown:
            raise ValueError(f"bet_lambdas: 含未知状态 {sorted(unknown)!r}")
        if missing:
            raise ValueError(
                f"bet_lambdas: 缺少状态的下注值 {sorted(missing)!r}"
            )
        out: dict[str, float] = {}
        for s in states:
            f = _finite_real(f"bet_lambdas[{s!r}]", bet_lambdas[s])
            if f < 0.0:
                raise ValueError(
                    f"bet_lambdas[{s!r}]: lambda 不得为负，得到 {f!r}"
                )
            out[s] = f
        return out

    @staticmethod
    def _validate_edges(edges, states, n_bins: int) -> dict[str, dict[int, str]]:
        if not isinstance(edges, dict):
            raise TypeError(f"edges: 需要 dict，实际得到 {type(edges).__name__}")
        unknown = set(edges) - set(states)
        missing = set(states) - set(edges)
        if unknown:
            raise ValueError(f"edges: 含未知状态 {sorted(unknown)!r}")
        if missing:
            raise ValueError(f"edges: 缺少状态的转移行 {sorted(missing)!r}")
        table: dict[str, dict[int, str]] = {}
        for s in states:
            row = edges[s]
            if not isinstance(row, dict):
                raise TypeError(
                    f"edges[{s!r}]: 需要 dict，实际得到 {type(row).__name__}"
                )
            if set(row) != set(range(n_bins)):
                raise ValueError(
                    f"edges[{s!r}]: 必须为每个反馈档位 0..{n_bins - 1} 给出下一状态，"
                    f"实际键为 {sorted(row)!r}"
                )
            parsed: dict[int, str] = {}
            for k, dst in row.items():
                if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < n_bins:
                    raise ValueError(
                        f"edges[{s!r}]: 档位键必须为 int 且位于 0..{n_bins - 1}，"
                        f"得到 {k!r}"
                    )
                if not isinstance(dst, str) or dst not in states:
                    raise ValueError(
                        f"edges[{s!r}][{k}]: 下一状态必须是 states 中的 str，"
                        f"得到 {dst!r}"
                    )
                parsed[k] = dst
            table[s] = parsed
        return table

    # ----- 属性与查表 -----------------------------------------------------

    @property
    def states(self) -> tuple[str, ...]:
        return self._states

    @property
    def initial_state(self) -> str:
        return self._initial_state

    @property
    def cutpoints(self) -> tuple[float, ...]:
        return self._cutpoints

    @property
    def num_bins(self) -> int:
        return len(self._cutpoints) - 1

    def lambda_for(self, state: str) -> float:
        """处于 state 时本轮必须封存的 lambda。"""
        return self._lambdas[state]

    def next_state_for(self, state: str, bin_index: int) -> str:
        """观察后按反馈档位查表得到下一状态。"""
        return self._edges[state][bin_index]

    def classify(self, x: float) -> int:
        """把已观察的 x ∈ [0,1] 按显式半开区间分入反馈档位。

        档位 k 为 [c_k, c_{k+1})，最后一档包含右端点 1。
        """
        last = self.num_bins - 1
        if x >= self._cutpoints[-1]:  # x == 1.0 归入最后一档
            return last
        lo, hi = 0, last
        while lo < hi:
            mid = (lo + hi) // 2
            if x < self._cutpoints[mid + 1]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    # ----- 序列化与指纹 ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "states": list(self._states),
            "initial_state": self._initial_state,
            "cutpoints": list(self._cutpoints),
            "bet_lambdas": {s: self._lambdas[s] for s in self._states},
            "edges": {
                s: {k: self._edges[s][k] for k in range(self.num_bins)}
                for s in self._states
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FiniteBettingStrategy":
        if not isinstance(data, dict):
            raise TypeError(
                f"strategy: 需要 dict，实际得到 {type(data).__name__}"
            )
        raw_edges = data.get("edges")
        edges = {}
        if isinstance(raw_edges, dict):
            # JSON 往返会把 int 档位键变成 str，这里统一转回 int。
            for state, row in raw_edges.items():
                if not isinstance(row, dict):
                    raise ValueError(f"strategy.edges[{state!r}]: 需要 dict")
                edges[state] = {}
                for k, dst in row.items():
                    if isinstance(k, str):
                        try:
                            ki = int(k)
                        except ValueError as exc:
                            raise ValueError(
                                f"strategy.edges[{state!r}]: 档位键 {k!r} 无法转 int"
                            ) from exc
                    elif isinstance(k, int) and not isinstance(k, bool):
                        ki = k
                    else:
                        raise ValueError(
                            f"strategy.edges[{state!r}]: 档位键 {k!r} 非法"
                        )
                    edges[state][ki] = dst
        else:
            edges = raw_edges
        return cls(
            data.get("states"),
            data.get("initial_state"),
            data.get("bet_lambdas"),
            edges,
            data.get("cutpoints"),
        )

    def fingerprint(self) -> str:
        """策略指纹：规范化 JSON 的 SHA3-256 十六进制摘要。

        策略只含 float/str/int 等 JSON 原生数据，序列化确定；恢复过程时
        策略指纹必须与封存进状态的指纹逐位一致（不允许换策略）。
        """
        import hashlib

        text = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha3_256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SealedBet:
    """一轮下注的封存记录（prepare 之后、observe 之前）。

    round_index : 将被观察的轮次编号（从 1 开始，等于封存时 t+1）
    state       : 封存时策略所处状态（只由已观察历史决定）
    lam         : 已封存的 lambda；本轮内没有任何更换入口
    """

    round_index: int
    state: str
    lam: float


class PredictableBettingProcess:
    """lambda 只依赖过去历史的自适应均值证据过程（封存式轮次接口）。

    参数
    ----
    m : 0 < m < 1，H0 条件均值上界。
    strategy : FiniteBettingStrategy
        仓库内有限状态表；每个状态封存一个 lambda（构造时静态校验
        0 <= lambda <= 1/m），观察 x 后按显式半开区间档位转移到下一状态。
    alpha : 0 < alpha < 1。

    每轮严格两步::

        bet = proc.prepare()   # 1) 只用已有历史查表，封存本轮 lambda
        snap = proc.observe(x) # 2) 再接收本轮 x，推进财富与状态

    - 重复 prepare 幂等：返回同一份封存，不换 lambda；
    - 未 prepare 不能 observe（RuntimeError）；
    - observe 之后封存解除，必须重新 prepare 下一轮；
    - 财富为单轨道 W_t = prod_{s<=t}(1 + lambda_s (x_s - m))，lambda_s 在
      看到 x_s 之前封存。H0 下逐因子条件期望 <= 1，W 是非负上鞅，Ville 界
      与 anytime p min(1, 1/max W) 成立。
    """

    def __init__(self, m, strategy: FiniteBettingStrategy, alpha):
        m_f = _finite_real("m", m)
        if not 0.0 < m_f < 1.0:
            raise ValueError(f"m: 必须严格位于 (0,1)，得到 {m_f!r}")
        a_f = _finite_real("alpha", alpha)
        if not 0.0 < a_f < 1.0:
            raise ValueError(f"alpha: 必须严格位于 (0,1)，得到 {a_f!r}")
        if not isinstance(strategy, FiniteBettingStrategy):
            raise TypeError(
                "strategy: 需要 FiniteBettingStrategy，实际得到 "
                f"{type(strategy).__name__}"
            )
        lam_max = 1.0 / m_f
        for s in strategy.states:
            lam = strategy.lambda_for(s)
            if lam > lam_max:
                raise ValueError(
                    f"strategy: 状态 {s!r} 封存 lambda={lam!r} 超过 1/m={lam_max!r}"
                )
        self._m = m_f
        self._alpha = a_f
        self._threshold = 1.0 / a_f
        self._log_threshold = math.log(self._threshold)
        self._strategy = strategy
        self._fingerprint = strategy.fingerprint()
        self._state = strategy.initial_state
        self._t = 0
        self._log_w = 0.0          # log W_0 = 0；归零后为 -inf 且永久保持
        self._max_log_w = 0.0
        self._stop_time: Optional[int] = None
        self._sealed: Optional[SealedBet] = None

    # ----- 属性 ------------------------------------------------------------

    @property
    def m(self) -> float:
        return self._m

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def t(self) -> int:
        return self._t

    @property
    def strategy(self) -> FiniteBettingStrategy:
        return self._strategy

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def state(self) -> str:
        """当前策略状态（下一轮 prepare 将使用的状态）。"""
        return self._state

    @property
    def sealed_round(self) -> Optional[int]:
        """当前封存的轮次；下一轮尚未 prepare 时为 None。"""
        return None if self._sealed is None else self._sealed.round_index

    def sealed_bet(self) -> Optional[SealedBet]:
        """当前封存记录；未封存返回 None。"""
        return self._sealed

    def log_wealth(self) -> float:
        """当前 log W；财富已归零为 -inf。"""
        return self._log_w

    # ----- 封存式轮次协议 --------------------------------------------------

    def prepare(self) -> SealedBet:
        """用已有历史（当前状态）查表，封存下一轮的 lambda。

        幂等：本轮已封存时重复调用返回同一份 SealedBet，lambda 不会改变。
        """
        if self._sealed is None:
            lam = self._strategy.lambda_for(self._state)
            self._sealed = SealedBet(
                round_index=self._t + 1,
                state=self._state,
                lam=lam,
            )
        return self._sealed

    def observe(self, x) -> EvidenceSnapshot:
        """接收本轮观察 x 并推进；必须先 prepare 封存 lambda。

        封存的 lambda 与状态在 x 校验之前即已确定；非法 x 不改变过程状态，
        封存保留（可换合法 x 再 observe）。
        """
        if self._sealed is None:
            raise RuntimeError(
                f"observe: 第 {self._t + 1} 轮尚未 prepare 封存 lambda，"
                "必须先下注再观察（不允许看见 x 后下注）"
            )
        x_f = _finite_real("x", x)
        if not 0.0 <= x_f <= 1.0:
            raise ValueError(f"x: 必须位于 [0,1]，得到 {x_f!r}")
        bet = self._sealed
        if self._log_w != -math.inf:
            factor = math.fma(bet.lam, x_f - self._m, 1.0)
            if factor <= 0.0:
                self._log_w = -math.inf
            else:
                self._log_w += math.log(factor)
        # 无论财富是否归零，反馈与状态转移照常（策略只依赖过去的 x）。
        bin_index = self._strategy.classify(x_f)
        self._state = self._strategy.next_state_for(bet.state, bin_index)
        self._t += 1
        self._sealed = None
        if self._log_w > self._max_log_w:
            self._max_log_w = self._log_w
        if self._stop_time is None and self._log_w >= self._log_threshold:
            self._stop_time = self._t
        return self.snapshot()

    def run_round(self, x) -> EvidenceSnapshot:
        """便利方法：prepare 后立即 observe。

        语义上仍是“先封存、后观察”：封存发生在读取 x 之前；x 非法时本轮
        封存保留（可换合法 x 再 observe），过程不推进。
        """
        self.prepare()
        return self.observe(x)

    def extend(self, xs) -> EvidenceSnapshot:
        """逐轮 prepare/observe 一个批次；先完整校验再推进（原子）。"""
        if isinstance(xs, (str, bytes)):
            raise TypeError(
                f"xs: 需要可迭代的实数序列，实际得到 {type(xs).__name__}"
            )
        try:
            materialized = list(xs)
        except TypeError as exc:
            raise TypeError(f"xs: 不可迭代: {type(xs).__name__}") from exc
        validated: list[float] = []
        for i, x in enumerate(materialized):
            f = _finite_real(f"xs[{i}]", x)
            if not 0.0 <= f <= 1.0:
                raise ValueError(f"xs[{i}]: 必须位于 [0,1]，得到 {f!r}")
            validated.append(f)
        if self._sealed is not None:
            raise RuntimeError(
                "extend: 当前轮已 prepare 封存但尚未 observe，请先完成本轮"
            )
        for x in validated:
            self.prepare()
            self.observe(x)
        return self.snapshot()

    # ----- 快照 ------------------------------------------------------------

    def snapshot(self) -> EvidenceSnapshot:
        if self._log_w == -math.inf:
            evidence = 0.0
        else:
            try:
                evidence = math.exp(self._log_w)
            except OverflowError:
                evidence = math.inf
        try:
            max_evidence = math.exp(self._max_log_w)
        except OverflowError:
            max_evidence = math.inf
        return EvidenceSnapshot(
            t=self._t,
            evidence=evidence,
            log_evidence=self._log_w,
            max_evidence=max_evidence,
            max_log_evidence=self._max_log_w,
            p_value=min(1.0, math.exp(-self._max_log_w)),
            threshold=self._threshold,
            stopped=self._stop_time is not None,
            stop_time=self._stop_time,
        )

    # ----- 序列化与恢复 ----------------------------------------------------

    def to_dict(self) -> dict:
        """导出状态：含策略指纹与当前封存轮次。

        完整策略表也一并存入（strategy）；fingerprint 为其摘要，恢复时重新
        计算并核对，任何对策略的改动都会使指纹不一致而拒绝恢复。
        """
        sealed = None
        if self._sealed is not None:
            sealed = {
                "round_index": self._sealed.round_index,
                "state": self._sealed.state,
                "lam": self._sealed.lam,
            }
        return {
            "version": PREDICTABLE_STATE_VERSION,
            "kind": "predictable_betting",
            "m": self._m,
            "alpha": self._alpha,
            "strategy": self._strategy.to_dict(),
            "fingerprint": self._fingerprint,
            "state": self._state,
            "t": self._t,
            "log_wealth": None if self._log_w == -math.inf else self._log_w,
            "max_log_wealth": self._max_log_w,
            "stop_time": self._stop_time,
            "sealed": sealed,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
    @classmethod
    def from_dict(
        cls,
        data: dict,
        *,
        strategy=_UNSET,
        m=_UNSET,
        alpha=_UNSET,
    ) -> "PredictableBettingProcess":
        """从 to_dict 的字典恢复。

        完整策略表从状态中重建，并以封存的策略指纹锁定：任何对策略的修改
        都会导致指纹不一致而抛 ValueError。可选地传入 strategy/m/alpha 作为
        额外断言（不一致即拒绝）。
        """
        if not isinstance(data, dict):
            raise TypeError(f"state: 需要 dict，实际得到 {type(data).__name__}")
        if data.get("version") != PREDICTABLE_STATE_VERSION:
            raise ValueError(
                f"state.version: 仅支持 {PREDICTABLE_STATE_VERSION}，"
                f"得到 {data.get('version')!r}"
            )
        if data.get("kind") != "predictable_betting":
            raise ValueError(
                f"state.kind: 需要 predictable_betting，得到 {data.get('kind')!r}"
            )

        m_s = _required_finite(data, "m")
        if not 0.0 < m_s < 1.0:
            raise ValueError(f"state.m: 越界 {m_s!r}")
        a_s = _required_finite(data, "alpha")
        if not 0.0 < a_s < 1.0:
            raise ValueError(f"state.alpha: 越界 {a_s!r}")

        strat_data = data.get("strategy")
        restored_strategy = FiniteBettingStrategy.from_dict(strat_data)
        fp = data.get("fingerprint")
        if not isinstance(fp, str):
            raise ValueError("state.fingerprint: 需要 str")
        actual_fp = restored_strategy.fingerprint()
        if fp != actual_fp:
            raise ValueError(
                "state.fingerprint: 与策略表不一致（策略被修改过），拒绝恢复"
            )
        if strategy is not _UNSET:
            if not isinstance(strategy, FiniteBettingStrategy):
                raise TypeError(
                    "strategy: 需要 FiniteBettingStrategy，实际得到 "
                    f"{type(strategy).__name__}"
                )
            if strategy.fingerprint() != fp:
                raise ValueError(
                    "strategy: 传入策略的指纹与封存状态不一致，恢复后不得换策略"
                )
        if m is not _UNSET and _finite_real("m", m) != m_s:
            raise ValueError(
                f"m: 恢复断言不一致（状态中 m={m_s!r}，给定 {float(m)!r}）"
            )
        if alpha is not _UNSET and _finite_real("alpha", alpha) != a_s:
            raise ValueError(
                f"alpha: 恢复断言不一致（状态中 alpha={a_s!r}，"
                f"给定 {float(alpha)!r}）"
            )
        lam_max = 1.0 / m_s
        for s in restored_strategy.states:
            lam = restored_strategy.lambda_for(s)
            if lam > lam_max:
                raise ValueError(
                    f"strategy: 状态 {s!r} 封存 lambda={lam!r} 超过 1/m={lam_max!r}"
                )

        t = data.get("t")
        if isinstance(t, bool) or not isinstance(t, int) or t < 0:
            raise ValueError(f"state.t: 需要非负 int，得到 {t!r}")
        cur_state = data.get("state")
        if not isinstance(cur_state, str) or cur_state not in restored_strategy.states:
            raise ValueError(f"state.state: 未知状态 {cur_state!r}")

        raw_lw = data.get("log_wealth")
        log_w = -math.inf if raw_lw is None else _finite_real(
            "state.log_wealth", raw_lw
        )
        max_log = _required_finite(data, "max_log_wealth")
        if max_log < 0.0:
            raise ValueError("state.max_log_wealth: 不得小于 0（W_0=1）")
        if max_log + 0.0 < log_w:
            raise ValueError("state: max_log_wealth 与 log_wealth 不一致")
        if t == 0 and max_log != 0.0:
            raise ValueError("state: t=0 时 max_log_wealth 必须恰为 0")

        stop = data.get("stop_time")
        if stop is not None:
            if isinstance(stop, bool) or not isinstance(stop, int) or not 1 <= stop <= t:
                raise ValueError(f"state.stop_time: 非法 {stop!r}")
        log_threshold = math.log(1.0 / a_s)
        if stop is not None and max_log < log_threshold:
            raise ValueError(
                "state: stop_time 与 max_log_wealth 不一致（未越界却已停止）"
            )

        sealed = cls._restore_sealed(
            data.get("sealed"), t, cur_state, restored_strategy, lam_max
        )

        obj = cls.__new__(cls)
        obj._m = m_s
        obj._alpha = a_s
        obj._threshold = 1.0 / a_s
        obj._log_threshold = log_threshold
        obj._strategy = restored_strategy
        obj._fingerprint = actual_fp
        obj._state = cur_state
        obj._t = t
        obj._log_w = log_w
        obj._max_log_w = max_log
        obj._stop_time = stop
        obj._sealed = sealed
        return obj

    @staticmethod
    def _restore_sealed(sealed_data, t, cur_state, restored_strategy, lam_max):
        if sealed_data is None:
            return None
        if not isinstance(sealed_data, dict):
            raise ValueError("state.sealed: 需要 dict 或 null")
        ri = sealed_data.get("round_index")
        st = sealed_data.get("state")
        lam = sealed_data.get("lam")
        if isinstance(ri, bool) or not isinstance(ri, int) or ri != t + 1:
            raise ValueError(
                f"state.sealed.round_index: 必须等于 t+1={t + 1}，得到 {ri!r}"
            )
        if not isinstance(st, str) or st != cur_state:
            raise ValueError(
                "state.sealed.state: 必须与当前状态 "
                f"{cur_state!r} 一致，得到 {st!r}"
            )
        lam_f = _finite_real("state.sealed.lam", lam)
        if lam_f != restored_strategy.lambda_for(st):
            raise ValueError("state.sealed.lam: 与策略表中该状态封存值不一致")
        if lam_f > lam_max:
            raise ValueError("state.sealed.lam: 超过 1/m")
        return SealedBet(ri, st, lam_f)

    @classmethod
    def from_json(cls, text: str, **assert_kwargs) -> "PredictableBettingProcess":
        """from_dict 的 JSON 字符串版本。"""
        if not isinstance(text, (str, bytes)):
            raise TypeError("text: 需要 str")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"state: JSON 解析失败: {exc}") from exc
        return cls.from_dict(data, **assert_kwargs)
