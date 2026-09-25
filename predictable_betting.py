"""只依赖过去的可预测下注（predictable betting）均值证据过程。

与 mean_evidence 的固定 lambda 网格不同，本模块允许每轮使用不同的
lambda_t，但强制 **先封存、后观察**：

1. 第 t 轮开始时只能依据已有历史 x_1..x_{t-1} 调用 prepare，
   策略依据“当前状态”（仅由上一轮的**离散反馈**转移而来）选出 lambda_t
   并封存；
2. 封存后本轮 lambda_t 不可更改，然后才允许调用 observe 接收 x_t；
3. x_t 到达后，按显式声明的、覆盖 [0,1] 且互不重叠的半开区间把 x_t
   分入一个离散反馈类别，策略据此转移到下一状态——lambda_t 从不依赖 x_t。

因此 lambda_t 是 F_{t-1} 可测的（predictable / 可预测过程）。财富

    W_t = prod_{s<=t} (1 + lambda_s (x_s - m)),   0 <= lambda_s <= 1/m

在 H0：E[x_t | F_{t-1}] <= m 下仍是初值 1 的非负上鞅：
E[1 + lambda_t(x_t-m) | F_{t-1}] <= 1。Ville 不等式照常给出
P_H0(存在 t: W_t >= 1/alpha) <= alpha，anytime p = min(1, 1/max_{s<=t} W_s)。

策略限定为仓库内的**有限状态表** FiniteStateStrategy：给定当前状态与
上一轮离散反馈类别，表中直接读出下一状态与下一轮 lambda。不执行任何
用户回调、可调用对象或字符串代码；序列化状态携带策略指纹，恢复后不得换表。

只依赖 Python 3.14 标准库。
"""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Optional, Sequence

from mean_evidence import _finite_real

__all__ = [
    "FeedbackBins",
    "FiniteStateStrategy",
    "PredictableBettingProcess",
    "BettingSnapshot",
    "BETTING_STATE_VERSION",
]

BETTING_STATE_VERSION = 1
_UNSET = object()


def _finite_real_from_state(data: dict, key: str) -> float:
    """从状态字典取必须存在的有限实数；缺失按 ValueError 处理。"""
    if key not in data:
        raise ValueError(f"state: 缺少字段 {key!r}")
    return _finite_real(f"state.{key}", data[key])


def _state_label(name: str, value) -> object:
    """状态标签允许非 bool 的 int 或 str；拒绝其他类型（含 float、bool）。"""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError(
            f"{name}: 状态标签需要 int（非 bool）或 str，"
            f"实际得到 {type(value).__name__}"
        )
    return value


# ---------------------------------------------------------------------------
# 显式半开区间反馈分类
# ---------------------------------------------------------------------------

class FeedbackBins:
    """覆盖 [0,1]、互不重叠的显式半开区间划分。

    由严格递增、互不相同且全部位于 (0,1) 的内部分点 cuts 定义：

        类别 0: [0, c_0)
        类别 k: [c_{k-1}, c_k)
        类别 n: [c_{n-1}, 1]

    即 x < 1 时区间左闭右开 [lo, hi)；右端点 1 归入最后一类。
    cuts 为空时只有一个类别（整个 [0,1]）。
    """

    __slots__ = ("_cuts",)

    def __init__(self, cuts: Sequence = ()):
        if isinstance(cuts, (str, bytes)) or not isinstance(cuts, Sequence):
            raise TypeError(
                "cuts: 需要有限实数序列（list/tuple），实际得到 "
                f"{type(cuts).__name__}"
            )
        prev = 0.0
        out: list[float] = []
        for i, c in enumerate(cuts):
            f = _finite_real(f"cuts[{i}]", c)
            if not 0.0 < f < 1.0:
                raise ValueError(f"cuts[{i}]: 分点必须严格位于 (0,1)，得到 {f!r}")
            if f <= prev:
                raise ValueError(
                    f"cuts[{i}]: 分点必须严格递增且互不重复，得到 {f!r}"
                    f"（前一个 {prev!r}）"
                )
            out.append(f)
            prev = f
        self._cuts: tuple[float, ...] = tuple(out)

    @property
    def cuts(self) -> tuple[float, ...]:
        return self._cuts

    @property
    def n_bins(self) -> int:
        """类别数 = 内部分点数 + 1，至少为 1。"""
        return len(self._cuts) + 1

    def classify(self, x) -> int:
        """把 x 属于 [0,1] 分入类别下标（区间左闭右开；x=1 归最后一类）。"""
        f = _finite_real("x", x)
        if not 0.0 <= f <= 1.0:
            raise ValueError(f"x: 必须位于 [0,1]，得到 {f!r}")
        if f == 1.0:
            return len(self._cuts)  # 右端点闭：1 属于最后一类
        # x < c_i 归第 i 类；恰等于分点 c_k 时落到右侧第 k+1 类。
        return bisect_right(self._cuts, f)

    def to_spec(self) -> list:
        return list(self._cuts)

    def __eq__(self, other) -> bool:
        return isinstance(other, FeedbackBins) and other._cuts == self._cuts

    def __repr__(self) -> str:
        return f"FeedbackBins(cuts={list(self._cuts)!r})"


# ---------------------------------------------------------------------------
# 有限状态表策略（无回调、无字符串代码）
# ---------------------------------------------------------------------------

class FiniteStateStrategy:
    """有限状态表：状态 + 上一轮离散反馈类别 -> 下一状态；状态 -> 本轮 lambda。

    参数
    ----
    m : float, 0 < m < 1
        H0 条件均值上界；表内所有 lambda 必须满足 0 <= lambda <= 1/m。
    states : 非空有限标签序列（int 非 bool，或 str），互不重复。
    initial_state : 必须属于 states；第 1 轮封存的 lambda 取自该状态。
    bins : FeedbackBins
        显式半开反馈区间划分。
    entries : 映射 state -> {"lambda": lam, "transitions": [q_0, ..., q_{B-1}]}
        每个状态对每个反馈类别都必须显式给出下一状态；缺失或类别数不符
        都按错误处理，不做任何默认填充。
    """

    def __init__(
        self,
        m,
        states: Sequence,
        initial_state,
        bins: FeedbackBins,
        entries: dict,
    ):
        m_f = _finite_real("m", m)
        if not 0.0 < m_f < 1.0:
            raise ValueError(f"m: 必须严格位于 (0,1)，得到 {m_f!r}")

        if isinstance(states, (str, bytes)) or not isinstance(states, Sequence):
            raise TypeError(
                f"states: 需要有限标签序列，实际得到 {type(states).__name__}"
            )
        labels: list = []
        seen = set()
        for i, lab in enumerate(states):
            lab = _state_label(f"states[{i}]", lab)
            if lab in seen:
                raise ValueError(f"states[{i}]: 状态标签重复 {lab!r}")
            seen.add(lab)
            labels.append(lab)
        if not labels:
            raise ValueError("states: 至少需要一个状态")

        initial = _state_label("initial_state", initial_state)
        if initial not in seen:
            raise ValueError(f"initial_state: {initial!r} 不在 states 中")

        if not isinstance(bins, FeedbackBins):
            raise TypeError(
                f"bins: 需要 FeedbackBins，实际得到 {type(bins).__name__}"
            )

        if not isinstance(entries, dict):
            raise TypeError(
                f"entries: 需要 dict 状态表，实际得到 {type(entries).__name__}"
            )

        lam_max = 1.0 / m_f
        n_bins = bins.n_bins
        table: dict = {}
        for lab in labels:
            if lab not in entries:
                raise ValueError(f"entries: 缺少状态 {lab!r} 的表项")
            entry = entries[lab]
            if not isinstance(entry, dict):
                raise TypeError(
                    f"entries[{lab!r}]: 需要 dict，实际得到 {type(entry).__name__}"
                )
            if "lambda" not in entry:
                raise ValueError(f"entries[{lab!r}]: 缺少 'lambda'")
            lam = _finite_real(f"entries[{lab!r}][lambda]", entry["lambda"])
            if not 0.0 <= lam <= lam_max:
                raise ValueError(
                    f"entries[{lab!r}][lambda]: 必须满足 0 <= lambda <= 1/m = "
                    f"{lam_max!r}，得到 {lam!r}"
                )
            trans = entry.get("transitions")
            if isinstance(trans, (str, bytes)) or not isinstance(trans, Sequence):
                raise TypeError(
                    f"entries[{lab!r}].transitions: 需要长度 {n_bins} 的序列，"
                    f"实际得到 {type(trans).__name__}"
                )
            if len(trans) != n_bins:
                raise ValueError(
                    f"entries[{lab!r}].transitions: 长度必须等于类别数 {n_bins}，"
                    f"得到 {len(trans)}"
                )
            next_states = []
            for k, q in enumerate(trans):
                q = _state_label(f"entries[{lab!r}].transitions[{k}]", q)
                if q not in seen:
                    raise ValueError(
                        f"entries[{lab!r}].transitions[{k}]: 目标状态 {q!r} "
                        "不在 states 中"
                    )
                next_states.append(q)
            table[lab] = {"lambda": lam, "transitions": tuple(next_states)}

        extra = set(entries) - seen
        if extra:
            raise ValueError(f"entries: 存在 states 之外的表项 {sorted(extra)!r}")

        self._m: float = m_f
        self._states: tuple = tuple(labels)
        self._initial = initial
        self._bins = bins
        self._table = table
        spec = {
            "m": m_f,
            "states": list(labels),
            "initial_state": initial,
            "cuts": bins.to_spec(),
            "entries": {
                lab: {
                    "lambda": table[lab]["lambda"],
                    "transitions": list(table[lab]["transitions"]),
                }
                for lab in labels
            },
        }
        canonical = json.dumps(
            spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        self._spec = spec
        self._fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def fixed(
        cls, m, lam, bins: Optional[FeedbackBins] = None
    ) -> "FiniteStateStrategy":
        """退化为固定 lambda 的单状态表：任何反馈都自环，每轮封存同一 lambda。"""
        b = bins if bins is not None else FeedbackBins(())
        return cls(
            m,
            (0,),
            0,
            b,
            {0: {"lambda": lam, "transitions": [0] * b.n_bins}},
        )

    @classmethod
    def from_spec(cls, spec: dict) -> "FiniteStateStrategy":
        """从 to_spec 重建策略（并重新计算、校验指纹）。"""
        if not isinstance(spec, dict):
            raise TypeError(f"strategy: 需要 dict，实际得到 {type(spec).__name__}")
        try:
            obj = cls(
                spec["m"],
                spec["states"],
                spec["initial_state"],
                FeedbackBins(spec["cuts"]),
                spec["entries"],
            )
        except KeyError as exc:
            raise ValueError(f"strategy: 缺少字段 {exc}") from exc
        return obj

    @property
    def m(self) -> float:
        return self._m

    @property
    def states(self) -> tuple:
        return self._states

    @property
    def initial_state(self):
        return self._initial

    @property
    def bins(self) -> FeedbackBins:
        return self._bins

    @property
    def fingerprint(self) -> str:
        """策略表（含 m、状态、分点、转移与 lambda）的 SHA-256 十六进制指纹。"""
        return self._fingerprint

    def lambda_at(self, state) -> float:
        return self._table[state]["lambda"]

    def next_state(self, state, feedback_bin: int):
        return self._table[state]["transitions"][feedback_bin]

    def to_spec(self) -> dict:
        """可 JSON 序列化的完整策略规格（普通 dict/list，与内部状态隔离）。"""
        return json.loads(json.dumps(self._spec, ensure_ascii=False, sort_keys=True))


@dataclass(frozen=True)
class BettingSnapshot:
    """可预测下注过程在某时刻的不可变快照。

    t                 : 已观察样本数（t=0 时 W_0 = 1）
    wealth            : W_t（极大时可能 inf，请优先使用 log_wealth）
    log_wealth        : log W_t；归零后恒为 -inf
    max_wealth        : max_{s<=t} W_s（含 W_0=1）
    max_log_wealth    : log(max_wealth)
    p_value           : min(1, 1/max_wealth)
    threshold         : 1/alpha
    stopped/stop_time : 首次 W_t >= 1/alpha 的时刻
    state             : 观察 x_t 并转移后的“下一状态”；t=0 为初始状态
    sealed_round      : 已封存、尚未观察的轮次号（t+1）；未封存为 None
    sealed_lambda     : 该封存轮次的 lambda_t；未封存为 None
    last_feedback     : 上一轮（第 t 轮）反馈类别；t=0 为 None
    """

    t: int
    wealth: float
    log_wealth: float
    max_wealth: float
    max_log_wealth: float
    p_value: float
    threshold: float
    stopped: bool
    stop_time: Optional[int]
    state: object
    sealed_round: Optional[int]
    sealed_lambda: Optional[float]
    last_feedback: Optional[int]


class PredictableBettingProcess:
    """逐轮 prepare -> observe 的可预测下注证据过程。

    协议（每轮严格两步，顺序不可颠倒）：

        proc.prepare()      # 仅依据历史封存本轮 lambda（重复调用幂等）
        proc.observe(x)     # 之后才能给出本轮 x；同一轮不能再次 observe

    未调用 prepare 直接 observe 抛 RuntimeError 且过程状态不变；封存后没有
    任何接口能在看到 x 后更换本轮 lambda。财富全程在 log 域累计，
    归零（因子恰为 0）后永久为零。
    """

    def __init__(self, strategy: FiniteStateStrategy, alpha):
        if not isinstance(strategy, FiniteStateStrategy):
            raise TypeError(
                "strategy: 需要 FiniteStateStrategy，实际得到 "
                f"{type(strategy).__name__}"
            )
        a_f = _finite_real("alpha", alpha)
        if not 0.0 < a_f < 1.0:
            raise ValueError(f"alpha: 必须严格位于 (0,1)，得到 {a_f!r}")

        self._strategy = strategy
        self._alpha = a_f
        self._threshold = 1.0 / a_f
        self._log_threshold = math.log(self._threshold)
        self._t: int = 0
        self._state = strategy.initial_state
        self._log_w: float = 0.0
        self._max_log_w: float = 0.0
        self._stop_time: Optional[int] = None
        self._last_feedback: Optional[int] = None
        self._sealed_round: Optional[int] = None
        self._sealed_lambda: Optional[float] = None

    @property
    def t(self) -> int:
        return self._t

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def strategy(self) -> FiniteStateStrategy:
        return self._strategy

    @property
    def fingerprint(self) -> str:
        """当前策略指纹；恢复后与封存状态中记录的指纹逐位一致。"""
        return self._strategy.fingerprint

    @property
    def state(self):
        """下一状态（prepare 将按它封存 lambda）；t=0 时为初始状态。"""
        return self._state

    @property
    def sealed_round(self) -> Optional[int]:
        """已封存、尚未观察的轮次号；观察完成后为 None。"""
        return self._sealed_round

    @property
    def sealed_lambda(self) -> Optional[float]:
        """封存轮次的 lambda；未封存时为 None。"""
        return self._sealed_lambda

    @property
    def last_feedback(self) -> Optional[int]:
        """上一轮观察落入的反馈类别；t=0 时为 None。"""
        return self._last_feedback

    def log_wealth(self) -> float:
        """当前 log 财富；归零后恒为 -inf。"""
        return self._log_w

    def snapshot(self) -> BettingSnapshot:
        log_w = self._log_w
        wealth = 0.0 if log_w == -math.inf else _safe_exp(log_w)
        max_wealth = _safe_exp(self._max_log_w)
        p_value = min(1.0, math.exp(-self._max_log_w))
        return BettingSnapshot(
            t=self._t,
            wealth=wealth,
            log_wealth=log_w,
            max_wealth=max_wealth,
            max_log_wealth=self._max_log_w,
            p_value=p_value,
            threshold=self._threshold,
            stopped=self._stop_time is not None,
            stop_time=self._stop_time,
            state=self._state,
            sealed_round=self._sealed_round,
            sealed_lambda=self._sealed_lambda,
            last_feedback=self._last_feedback,
        )

    def prepare(self) -> float:
        """为下一轮（第 t+1 轮）封存 lambda；幂等。

        只读取已有历史（体现为当前状态）：lambda 由有限状态表对当前状态
        直接读出。已封存且尚未 observe 时重复调用返回同一 lambda，
        封存轮次与取值都不改变。
        """
        if self._sealed_round is None:
            self._sealed_lambda = self._strategy.lambda_at(self._state)
            self._sealed_round = self._t + 1
        return self._sealed_lambda

    def observe(self, x) -> BettingSnapshot:
        """接收本轮观察 x；必须且只能在本轮 prepare 之后调用一次。

        未 prepare 抛 RuntimeError；同一轮重复 observe 也抛 RuntimeError，
        且过程状态保持不变。x 的校验先于任何状态推进，非法 x 不留变更。
        """
        if self._sealed_round is None:
            raise RuntimeError(
                "observe: 本轮尚未 prepare；必须先用已有历史封存 lambda，"
                "禁止看到当前样本后下注"
            )
        f = _finite_real("x", x)
        if not 0.0 <= f <= 1.0:
            raise ValueError(f"x: 必须位于 [0,1]，得到 {f!r}")

        lam = self._sealed_lambda
        round_no = self._sealed_round

        if self._log_w != -math.inf:
            factor = math.fma(lam, f - self._strategy.m, 1.0)
            if factor <= 0.0:
                self._log_w = -math.inf
            else:
                self._log_w = self._log_w + math.log(factor)
        self._t = round_no

        bin_idx = self._strategy.bins.classify(f)
        self._last_feedback = bin_idx
        self._state = self._strategy.next_state(self._state, bin_idx)

        if self._log_w > self._max_log_w:
            self._max_log_w = self._log_w
        if self._stop_time is None and self._log_w >= self._log_threshold:
            self._stop_time = round_no
        self._sealed_round = None
        self._sealed_lambda = None
        return self.snapshot()
