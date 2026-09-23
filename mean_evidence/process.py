"""可选停止下的均值证据过程（核心）。

数学对象
--------

观测 ``x_t ∈ [0,1]``，零假设为条件均值 ``E[x_t | F_{t-1}] <= m``
（``m ∈ (0,1)``）。对固定网格 ``0 <= λ_k <= 1/m`` 定义财富过程

    W_k,t = Π_{s<=t} (1 + λ_k (x_s - m)),

对网格均匀混合得到证据过程

    E_t = (1/K) Σ_k W_k,t （E_0 = 1）。

当某因子为 0 时对应财富永久归零（此后恒为 0）。

在零假设下，每个因子的条件期望

    E[1 + λ(x_t-m) | F_{t-1}] = 1 + λ(E[x_t|F_{t-1}] - m) <= 1，

故 ``E[W_k,t] <= 1``、``E[E_t] <= 1``，且在可选停止（停时）下
Ville 不等式给出 ``P(存在 t: E_t >= 1/α) <= α``。

注意：该保证只需要零假设的**条件均值**逐期成立（外加可测的停时），
不要求独立性；本库只对这一指定假设下的证据负责，不对外推广任何
更强结论。

实现要点
--------

* 全部量在 log 域维护：``log W = Σ log factor``，logsumexp 混合，
  比较 ``E_t >= 1/α`` 等价为 ``log E_t >= -log α``，因此财富即便
  超过 ``sys.float_info.max`` 也不影响停止时刻与 p 值判定。
* 因子恰为 0（``factor == 0.0``）时该财富标记死亡，log 权记 -inf。
* :class:`MeanEvidenceProcess` 支持逐点/批量追加与状态序列化；
  恢复时 ``m``、lambda 网格、``alpha`` 与状态不一致会被拒绝。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Final

from .errors import (
    EvidenceError,
    InvalidParameterError,
    SerializationError,
    StateMismatchError,
)
from .validation import finite_observation, finite_real, nonempty_finite_grid

__all__ = ["MeanEvidenceProcess", "AnalysisResult", "analyze"]

_STATE_VERSION: Final = 1
#: 浮点边界容差：λ(0-m) 这一项在 λ=1/m 附近可能因两次舍入略小于 0。
#: 数学上该处因子本就为 0（财富归零），在此幅度内按零因子处理。
_NEG_FACTOR_TOL: Final = 8.0 * math.ulp(1.0)  # ≈ 1.78e-15
_NEG_INF: Final = -math.inf


def _validated_m(m: Any) -> float:
    f = finite_real("m", m)
    if not 0.0 < f < 1.0:
        raise InvalidParameterError(f"m 必须满足 0 < m < 1（收到 {m!r}）")
    return f


def _validated_grid(m: float, lambda_grid: Any) -> list[float]:
    grid = nonempty_finite_grid(lambda_grid)
    inv_m = 1.0 / m
    for i, lam in enumerate(grid):
        if not 0.0 <= lam <= inv_m:
            raise InvalidParameterError(
                f"lambda_grid[{i}]={lam!r} 越界：必须满足 0 <= lambda <= 1/m"
                f"（m={m!r}，1/m={inv_m!r}）"
            )
    return grid


def _validated_alpha(alpha: Any) -> float:
    f = finite_real("alpha", alpha)
    if not 0.0 < f < 1.0:
        raise InvalidParameterError(f"alpha 必须满足 0 < alpha < 1（收到 {alpha!r}）")
    return f


def _logsumexp_active(values: list[float]) -> float:
    """活跃（非 -inf）log 权的 logsumexp；调用方保证列表非空。"""
    top = max(values)
    # top 有限：单次 log factor <= log 2，有限步数累加在实际问题中恒有限。
    return top + math.log(sum(math.exp(v - top) for v in values))


@dataclass(frozen=True)
class AnalysisResult:
    """一次性分析结果。"""

    t: int
    log_evidence: float
    p_value: float
    stop_time: int | None
    alpha: float

    @property
    def rejected(self) -> bool:
        """是否已在某一时刻首次越过 1/alpha。"""
        return self.stop_time is not None


class MeanEvidenceProcess:
    """有状态的均值证据过程。

    参数
    ----
    m:
        零假设条件均值上界，``0 < m < 1``。
    lambda_grid:
        非空有限数值序列，每项满足 ``0 <= λ <= 1/m``。网格在构造后
        即固定，不可更改。
    alpha:
        检验水平，``0 < alpha < 1``；拒绝阈值为 ``1/alpha``。

    非法输入抛出 :class:`InvalidParameterError`（消息含参数名），
    不做静默纠正。
    """

    __slots__ = (
        "_m",
        "_grid",
        "_alpha",
        "_log_threshold",
        "_t",
        "_log_wealth",
        "_dead",
        "_max_log_evidence",
        "_stop_time",
    )

    def __init__(self, m: Any, lambda_grid: Any, alpha: Any) -> None:
        self._m = _validated_m(m)
        self._grid = _validated_grid(self._m, lambda_grid)
        self._alpha = _validated_alpha(alpha)
        self._log_threshold = -math.log(self._alpha)
        self._reset_state()

    def _reset_state(self) -> None:
        self._t = 0
        self._log_wealth = [0.0] * len(self._grid)
        self._dead = [False] * len(self._grid)
        # max_{s<=t} log E_s；E_0 = 1，故初值 0。
        self._max_log_evidence = 0.0
        self._stop_time: int | None = None

    # ------------------------------------------------------------------ #
    # 只读属性
    # ------------------------------------------------------------------ #
    @property
    def m(self) -> float:
        return self._m

    @property
    def lambda_grid(self) -> tuple[float, ...]:
        return tuple(self._grid)

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def t(self) -> int:
        return self._t

    @property
    def stop_time(self) -> int | None:
        """首次 ``E_t >= 1/alpha`` 的时刻（从 1 开始计），未越界为 None。"""
        return self._stop_time

    @property
    def rejected(self) -> bool:
        return self._stop_time is not None

    @property
    def log_wealth(self) -> tuple[float, ...]:
        """各 λ 当前的 log W；已归零的财富为 ``-inf``。"""
        return tuple(
            _NEG_INF if dead else lw for lw, dead in zip(self._log_wealth, self._dead)
        )

    @property
    def log_evidence(self) -> float:
        """当前 ``log E_t``（E_0 = 1）；全部财富归零时为 ``-inf``。"""
        active = [lw for lw, dead in zip(self._log_wealth, self._dead) if not dead]
        if not active:
            return _NEG_INF
        return _logsumexp_active(active) - math.log(len(self._grid))

    @property
    def evidence(self) -> float:
        """当前混合财富 ``E_t``。

        log 域之外的直流量：``E_t`` 超过 ``float_info.max`` 时返回
        ``inf``（停止时刻与 p 值在 log 域判定，不受影响）；全部归零
        时返回 0.0。
        """
        le = self.log_evidence
        if le == _NEG_INF:
            return 0.0
        try:
            return math.exp(le)
        except OverflowError:
            return math.inf

    @property
    def max_log_evidence(self) -> float:
        return self._max_log_evidence

    @property
    def p_value(self) -> float:
        """anytime p 值 ``min(1, 1/max_{s<=t} E_s)``（当前时刻口径）。

        下溢为 0.0 即正确舍入结果。
        """
        return math.exp(-self._max_log_evidence)

    # ------------------------------------------------------------------ #
    # 状态推进
    # ------------------------------------------------------------------ #
    def append(self, x: Any) -> float:
        """追加一个观测，返回更新后的 ``log E_t``。

        与 :meth:`extend` 同样原子：更新中途失败则状态完全不变。
        """
        fx = finite_observation(x)
        if not 0.0 <= fx <= 1.0:
            _raise_observation_error(fx)
        return self._update_many_atomic([fx])

    def extend(self, xs: Any) -> float:
        """批量追加观测，返回更新后的 ``log E_t``。

        先对整批输入做校验，再在快照上推进；任何一步失败都整体回滚，
        不会留下部分更新。
        """
        checked: list[float] = []
        if isinstance(xs, (str, bytes)) or not hasattr(xs, "__iter__"):
            raise InvalidParameterError(
                f"extend 需要有限数值序列（收到 {type(xs).__name__}）"
            )
        for i, x in enumerate(xs):
            fx = finite_observation(x)
            if not 0.0 <= fx <= 1.0:
                _raise_observation_error(fx, index=i)
            checked.append(fx)
        return self._update_many_atomic(checked)

    def _update_many_atomic(self, checked: list[float]) -> float:
        snapshot = self._snapshot_state()
        try:
            for fx in checked:
                self._update_one(fx)
        except OverflowError:
            # log 域加法在实际长度下不会溢出；保留回滚以严格满足
            # “失败不留部分变更”，并显式区别于正常无解结论。
            self._restore_state(snapshot)
            raise EvidenceError(
                "内部数值上溢，状态已回滚（输入规模超出 float 可表示范围）"
            )
        except EvidenceError:
            # 内部不变量破坏等：同样回滚，原错误继续上抛。
            self._restore_state(snapshot)
            raise
        return self.log_evidence

    def _update_one(self, x: float) -> None:
        m = self._m
        d = x - m
        for k, lam in enumerate(self._grid):
            if self._dead[k]:
                continue
            factor = 1.0 + lam * d
            if factor <= 0.0:
                if factor < -_NEG_FACTOR_TOL:
                    # 校验已保证 0 <= λ <= 1/m 与 x ∈ [0,1]，走到这里
                    # 意味着浮点边界被远超，属于内部不变量破坏。
                    raise EvidenceError(
                        f"内部不变量破坏：lambda_grid[{k}]={lam!r} 与 x={x!r} "
                        f"产生负因子 {factor!r}，超出边界舍入容差"
                    )
                # 0.0 或边界舍入的微小负值：数学上因子为 0，财富归零。
                self._dead[k] = True
                self._log_wealth[k] = _NEG_INF
                continue
            self._log_wealth[k] += math.log(factor)
        self._t += 1

        log_e = self.log_evidence
        if log_e > self._max_log_evidence:
            self._max_log_evidence = log_e
        if self._stop_time is None and log_e >= self._log_threshold:
            self._stop_time = self._t

    # ------------------------------------------------------------------ #
    # 序列化
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """导出可 JSON 化的纯数据状态。"""
        return {
            "version": _STATE_VERSION,
            "m": self._m,
            "lambda_grid": list(self._grid),
            "alpha": self._alpha,
            "t": self._t,
            "log_wealth": [
                None if dead else lw
                for lw, dead in zip(self._log_wealth, self._dead)
            ],
            "dead": list(self._dead),
            "max_log_evidence": self._max_log_evidence,
            "stop_time": self._stop_time,
        }

    def to_json(self) -> str:
        """序列化为 JSON 字符串（拒绝 NaN/Infinity 常量）。"""
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True)

    @classmethod
    def from_dict(cls, state: Any, *, m: Any = None, lambda_grid: Any = None,
                  alpha: Any = None) -> "MeanEvidenceProcess":
        """从 :meth:`to_dict` 的状态恢复。

        必须显式传入与状态一致的 ``m`` / ``lambda_grid`` / ``alpha``
        （浮点逐一严格相等）；任一不一致抛 :class:`StateMismatchError`。
        恢复后不允许更换 m 或网格——状态结构本身也携带原参数，
        两者必须对得上。
        """
        if not isinstance(state, dict):
            raise SerializationError(f"状态必须为 dict（收到 {type(state).__name__}）")
        if state.get("version") != _STATE_VERSION:
            raise SerializationError(
                f"不支持的状态版本（收到 {state.get('version')!r}，"
                f"期望 {_STATE_VERSION}）"
            )
        def _strict_state_float(name: str, value: Any) -> float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SerializationError(f"状态字段 {name} 必须为数值（收到 {value!r}）")
            f = float(value)
            if not math.isfinite(f):
                raise SerializationError(f"状态字段 {name} 必须有限（收到 {value!r}）")
            return f

        try:
            saved_m = _strict_state_float("m", state["m"])
            raw_grid = state["lambda_grid"]
            if not isinstance(raw_grid, list):
                raise SerializationError("状态字段 lambda_grid 必须为列表")
            saved_grid = [_strict_state_float(f"lambda_grid[{i}]", v)
                          for i, v in enumerate(raw_grid)]
            saved_alpha = _strict_state_float("alpha", state["alpha"])
            saved_t = state["t"]
            saved_lw = state["log_wealth"]
            saved_dead = state["dead"]
            saved_max = _strict_state_float("max_log_evidence",
                                            state["max_log_evidence"])
            saved_stop = state["stop_time"]
        except KeyError as exc:
            raise SerializationError(f"状态字段缺失：{exc}") from exc
        except (TypeError, ValueError) as exc:
            raise SerializationError(f"状态字段类型错误：{exc}") from exc

        if m is None or lambda_grid is None or alpha is None:
            raise InvalidParameterError(
                "恢复状态必须显式传入 m、lambda_grid、alpha 以核对身份"
            )
        m = _validated_m(m)
        grid = _validated_grid(m, lambda_grid)
        alpha = _validated_alpha(alpha)
        if m != saved_m:
            raise StateMismatchError(
                f"m 与状态不一致：传入 {m!r}，状态记录 {saved_m!r}；恢复后不得更改 m"
            )
        if grid != saved_grid:
            raise StateMismatchError(
                "lambda_grid 与状态不一致；恢复后不得更改网格"
                f"（传入 {grid!r}，状态记录 {saved_grid!r}）"
            )
        if alpha != saved_alpha:
            raise StateMismatchError(
                f"alpha 与状态不一致：传入 {alpha!r}，状态记录 {saved_alpha!r}"
            )

        proc = cls.__new__(cls)
        proc._m = m
        proc._grid = grid
        proc._alpha = alpha
        proc._log_threshold = -math.log(alpha)
        proc._load_validated_state(saved_t, saved_lw, saved_dead, saved_max, saved_stop)
        return proc

    @classmethod
    def from_json(cls, text: Any, *, m: Any = None, lambda_grid: Any = None,
                  alpha: Any = None) -> "MeanEvidenceProcess":
        """从 :meth:`to_json` 字符串恢复，规则同 :meth:`from_dict`。"""
        if not isinstance(text, str):
            raise SerializationError(f"JSON 必须为 str（收到 {type(text).__name__}）")

        def _reject_constant(value: str) -> Any:
            raise SerializationError(f"JSON 中不得出现 {value} 常量")

        try:
            state = json.loads(text, parse_constant=_reject_constant)
        except json.JSONDecodeError as exc:
            raise SerializationError(f"JSON 解析失败：{exc}") from exc
        return cls.from_dict(state, m=m, lambda_grid=lambda_grid, alpha=alpha)

    def _load_validated_state(self, t: Any, lw: Any, dead: Any,
                              max_log_evidence: Any, stop_time: Any) -> None:
        if isinstance(t, bool) or not isinstance(t, int) or t < 0:
            raise SerializationError(f"t 必须为非负 int（收到 {t!r}）")
        k = len(self._grid)
        if not isinstance(lw, list) or len(lw) != k:
            raise SerializationError("log_wealth 必须为与网格等长的列表")
        if not isinstance(dead, list) or len(dead) != k:
            raise SerializationError("dead 必须为与网格等长的列表")
        log_wealth: list[float] = []
        deads: list[bool] = []
        for i, (entry, is_dead) in enumerate(zip(lw, dead)):
            if not isinstance(is_dead, bool):
                raise SerializationError(f"dead[{i}] 必须为 bool")
            if is_dead:
                if entry is not None:
                    raise SerializationError(f"log_wealth[{i}] 死亡时必须为 null")
                log_wealth.append(_NEG_INF)
                deads.append(True)
            else:
                if isinstance(entry, bool) or not isinstance(entry, (int, float)):
                    raise SerializationError(f"log_wealth[{i}] 必须为数值或 null")
                f = float(entry)
                if not math.isfinite(f):
                    raise SerializationError(f"log_wealth[{i}] 必须有限（死亡用 null 表示）")
                # 每步因子最大为 1 + (1/m-1)*1 = 1/m，故 log 上界 t*log(1/m)。
                bound = t * math.log(1.0 / self._m) + 1e-9
                if f > bound:
                    raise SerializationError(
                        f"log_wealth[{i}]={f} 超过 t={t} 步的理论上界 t*log(1/m)"
                    )
                log_wealth.append(f)
                deads.append(False)
        if not math.isfinite(max_log_evidence) or max_log_evidence < 0.0:
            raise SerializationError("max_log_evidence 必须为非负有限数")
        if stop_time is not None:
            if isinstance(stop_time, bool) or not isinstance(stop_time, int):
                raise SerializationError("stop_time 必须为 int 或 null")
            if not 1 <= stop_time <= t:
                raise SerializationError("stop_time 必须满足 1 <= stop_time <= t")

        # 内部一致性：运行最大值不得低于当前 log E_t；停止标志与阈值必须对得上。
        active = [v for v in log_wealth if math.isfinite(v)]
        cur_log_e = (
            _NEG_INF if not active
            else _logsumexp_active(active) - math.log(k)
        )
        tol = 1e-12
        if max_log_evidence + tol < cur_log_e:
            raise SerializationError(
                f"max_log_evidence={max_log_evidence} 小于当前 log E_t={cur_log_e}"
            )
        if stop_time is not None and max_log_evidence + tol < self._log_threshold:
            raise SerializationError("stop_time 非空但 max_log_evidence 未达到 -log(alpha)")
        if stop_time is None and max_log_evidence > self._log_threshold + tol:
            # 超过阈值却没有停止时刻，说明历史上必然该停——状态自相矛盾。
            raise SerializationError("max_log_evidence 已超过 -log(alpha) 却无 stop_time")

        self._t = t
        self._log_wealth = log_wealth
        self._dead = deads
        self._max_log_evidence = float(max_log_evidence)
        self._stop_time = stop_time

    def _snapshot_state(self) -> tuple[int, list[float], list[bool], float, int | None]:
        return (
            self._t,
            list(self._log_wealth),
            list(self._dead),
            self._max_log_evidence,
            self._stop_time,
        )

    def _restore_state(self, snap: tuple) -> None:
        self._t, lw, dead, mle, stop = snap
        self._log_wealth = list(lw)
        self._dead = list(dead)
        self._max_log_evidence = mle
        self._stop_time = stop


def _raise_observation_error(fx: float, index: int | None = None) -> None:
    from .errors import InvalidObservationError

    where = f"xs[{index}]" if index is not None else "x"
    raise InvalidObservationError(f"{where}={fx!r} 越界：必须满足 0 <= x <= 1")


def analyze(m: Any, lambda_grid: Any, observations: Any, alpha: Any) -> AnalysisResult:
    """一次性便捷分析：构造过程并喂入全部观测。

    观测序列在任何更新前完成整批校验，失败时无状态被创建。
    """
    proc = MeanEvidenceProcess(m, lambda_grid, alpha)
    proc.extend(observations)
    return AnalysisResult(
        t=proc.t,
        log_evidence=proc.log_evidence,
        p_value=proc.p_value,
        stop_time=proc.stop_time,
        alpha=proc.alpha,
    )
