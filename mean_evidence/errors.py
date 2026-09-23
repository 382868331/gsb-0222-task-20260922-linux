"""异常类型。

所有由本库主动抛出的错误都继承 :class:`EvidenceError`
（同时也是 :class:`ValueError`），错误消息一律携带出问题的参数名，
便于调用方定位。本库不做静默纠正。
"""

from __future__ import annotations


class EvidenceError(ValueError):
    """本库所有公开错误的基类。"""


class InvalidParameterError(EvidenceError):
    """构造参数（m / lambda 网格 / alpha / 反序列化期望参数）非法。"""


class InvalidObservationError(EvidenceError):
    """观测值 x 非法（非有限实数、越界或类型错误）。"""


class StateMismatchError(EvidenceError):
    """用与序列化状态不一致的 m / lambda 网格 / alpha 恢复状态。"""


class SerializationError(EvidenceError):
    """序列化数据结构损坏、版本不符或含非法 JSON 常量。"""
