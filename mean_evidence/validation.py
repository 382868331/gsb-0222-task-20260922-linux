"""输入校验工具。

统一契约：

* 数值输入拒绝 ``NaN`` / ``Infinity`` / ``-Infinity``；
* 布尔值 ``True`` / ``False`` 不冒充数值或整数（Python 中 ``bool``
  是 ``int`` 的子类，必须显式拦截）；
* 非法输入抛出带参数名的 :class:`InvalidParameterError` 或
  :class:`InvalidObservationError`，绝不静默纠正。
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from .errors import InvalidObservationError, InvalidParameterError


def reject_bool(name: str, value: Any) -> None:
    if isinstance(value, bool):
        raise InvalidParameterError(f"{name} 不得为 bool（收到 {value!r}）")


def finite_real(name: str, value: Any) -> float:
    """校验单个有限实数值，返回 ``float``。"""
    reject_bool(name, value)
    if isinstance(value, int):
        f = float(value)
    elif isinstance(value, float):
        f = value
    else:
        raise InvalidParameterError(
            f"{name} 必须为 int 或 float（收到 {type(value).__name__}）"
        )
    if not math.isfinite(f):
        raise InvalidParameterError(f"{name} 必须为有限实数，拒绝 NaN/Infinity（收到 {value!r}）")
    return f


def finite_observation(x: Any) -> float:
    """校验单个观测：有限实数。"""
    if isinstance(x, bool):
        raise InvalidObservationError(f"x 不得为 bool（收到 {x!r}）")
    if isinstance(x, int):
        f = float(x)
    elif isinstance(x, float):
        f = x
    else:
        raise InvalidObservationError(
            f"x 必须为 int 或 float（收到 {type(x).__name__}）"
        )
    if not math.isfinite(f):
        raise InvalidObservationError(f"x 必须为有限实数，拒绝 NaN/Infinity（收到 {x!r}）")
    return f


def nonempty_finite_grid(values: Any) -> list[float]:
    """校验 lambda 网格：非空有限可迭代对象，全部为有限实数。

    同时拒绝把单个标量误传为网格、以及对网格的惰性迭代器造成的
    “校验通过后再迭代失败”——这里一次性物化为列表。
    """
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidParameterError(
            "lambda_grid 必须为非空有限可迭代数值序列"
            f"（收到 {type(values).__name__}）"
        )
    out: list[float] = []
    for i, v in enumerate(values):
        out.append(finite_real(f"lambda_grid[{i}]", v))
    if not out:
        raise InvalidParameterError("lambda_grid 不得为空")
    return out
