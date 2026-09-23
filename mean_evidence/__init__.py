"""mean_evidence —— 可选停止下的均值证据过程。

公开接口::

    from mean_evidence import (
        MeanEvidenceProcess,   # 有状态过程：追加观测 / 序列化恢复
        AnalysisResult,        # analyze 的只读结果
        analyze,               # 一次性便捷分析
        EvidenceError,         # 本库所有错误的基类
        InvalidParameterError,
        InvalidObservationError,
        StateMismatchError,
        SerializationError,
    )

只依赖 Python 3.14 标准库。统计结论的适用范围见
:mod:`mean_evidence.process` 模块文档：证据仅针对零假设的**条件均值**
``E[x_t | F_{t-1}] <= m`` 逐期成立这一前提（停时可测即可，不要求
独立性），不对外推广更强结论。
"""

from __future__ import annotations

from .errors import (
    EvidenceError,
    InvalidObservationError,
    InvalidParameterError,
    SerializationError,
    StateMismatchError,
)
from .process import AnalysisResult, MeanEvidenceProcess, analyze

__all__ = [
    "MeanEvidenceProcess",
    "AnalysisResult",
    "analyze",
    "EvidenceError",
    "InvalidParameterError",
    "InvalidObservationError",
    "StateMismatchError",
    "SerializationError",
]

__version__ = "1.0.0"
