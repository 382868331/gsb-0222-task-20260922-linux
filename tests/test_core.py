"""核心行为：边界 0/1、零因子、log 域混合、p 值、可选停止。

数值期望值由测试内直接按定义用 ``math`` 独立重算（不经过被测核心），
Bernoulli 期望的独立精确参考见 test_bernoulli_enumeration.py。
"""

from __future__ import annotations

import math
import unittest

from mean_evidence import (
    AnalysisResult,
    InvalidObservationError,
    InvalidParameterError,
    MeanEvidenceProcess,
    analyze,
)

GRID = [0.0, 0.5, 1.0, 2.0]  # m=0.5 时 1/m = 2，恰好含端点
M = 0.5
ALPHA = 0.2


def _reference_evidence(m: float, grid: list[float], xs: list[float]) -> float:
    """直接按定义计算 E_t，独立于核心实现。"""
    total = 0.0
    for lam in grid:
        w = 1.0
        for x in xs:
            w *= 1.0 + lam * (x - m)
        total += w
    return total / len(grid)


class CoreBehaviorTests(unittest.TestCase):
    def test_initial_state(self) -> None:
        p = MeanEvidenceProcess(M, GRID, ALPHA)
        self.assertEqual(p.t, 0)
        self.assertEqual(p.evidence, 1.0)
        self.assertEqual(p.log_evidence, 0.0)
        self.assertEqual(p.p_value, 1.0)
        self.assertIsNone(p.stop_time)
        self.assertFalse(p.rejected)
        self.assertEqual(p.log_wealth, (0.0, 0.0, 0.0, 0.0))

    def test_boundary_observations_zero_and_one(self) -> None:
        # 边界 0 与 1 必须被接受，并与独立定义逐项吻合。
        for xs in ([0.0], [1.0], [0.0, 1.0, 1.0, 0.0], [1, 0, 1]):
            with self.subTest(xs=xs):
                p = MeanEvidenceProcess(M, GRID, ALPHA)
                p.extend(xs)
                ref = _reference_evidence(M, GRID, [float(x) for x in xs])
                self.assertTrue(math.isclose(p.evidence, ref, rel_tol=1e-14, abs_tol=1e-14))
                self.assertTrue(math.isclose(p.log_evidence, math.log(ref), rel_tol=1e-14))

    def test_integer_observations_accepted(self) -> None:
        p = MeanEvidenceProcess(M, GRID, ALPHA)
        p.append(1)
        p.extend([0, 1])
        self.assertEqual(p.t, 3)

    def test_zero_factor_kills_wealth_permanently(self) -> None:
        # lambda=2=1/m，x=0：因子 1 + 2*(0-0.5) = 0。
        p = MeanEvidenceProcess(M, GRID, ALPHA)
        p.append(0.0)
        lw = p.log_wealth
        self.assertEqual(lw[3], -math.inf)
        self.assertTrue(math.isfinite(lw[0]))
        # 死亡后再观测，该项保持 -inf，不“复活”。
        p.extend([1.0, 1.0, 1.0])
        self.assertEqual(p.log_wealth[3], -math.inf)
        # 手工定义核对其余三项与混合值。
        ref = _reference_evidence(M, GRID, [0.0, 1.0, 1.0, 1.0])
        self.assertTrue(math.isclose(p.evidence, ref, rel_tol=1e-14))

    def test_all_wealth_zero_gives_zero_evidence(self) -> None:
        # 网格只含端点 lambda=1/m：一个 x=0 即全部归零。
        p = MeanEvidenceProcess(M, [2.0], ALPHA)
        p.append(0.0)
        self.assertEqual(p.evidence, 0.0)
        self.assertEqual(p.log_evidence, -math.inf)
        self.assertEqual(p.p_value, 1.0)  # max E_s 仍为 E_0 = 1
        p.append(1.0)  # 归零后继续喂数据不报错也不复活
        self.assertEqual(p.evidence, 0.0)
        self.assertIsNone(p.stop_time)

    def test_lambda_zero_is_cash_neutral(self) -> None:
        p = MeanEvidenceProcess(M, [0.0], ALPHA)
        p.extend([0.0, 1.0, 0.3])
        self.assertEqual(p.evidence, 1.0)
        self.assertEqual(p.log_evidence, 0.0)

    def test_optional_stopping_and_pvalue(self) -> None:
        # 构造持续高于 m 的序列，必然越过 1/alpha=5。
        p = MeanEvidenceProcess(M, GRID, ALPHA)
        stop = None
        for t in range(1, 60):
            p.append(0.9)
            if p.stop_time is not None:
                stop = t
                break
        self.assertIsNotNone(stop)
        self.assertTrue(p.evidence >= 1.0 / ALPHA)
        # 首次越界后再追加，stop_time 锁定在首次。
        first = p.stop_time
        p.extend([0.9, 0.9])
        self.assertEqual(p.stop_time, first)
        # p 值 = 1/历史最大 E，且 <= alpha、单调不增、<=1。
        self.assertLessEqual(p.p_value, ALPHA + 1e-15)
        self.assertLessEqual(p.p_value, 1.0)
        self.assertTrue(
            math.isclose(p.p_value, math.exp(-p.max_log_evidence), rel_tol=1e-15)
        )

    def test_p_value_is_running_maximum(self) -> None:
        # 先冲高再回落：p 值锚定历史最大证据，不随当前 E 反弹。
        p = MeanEvidenceProcess(0.5, [0.0, 1.0, 2.0], 0.05)
        highs = []
        p.extend([0.95] * 8)
        highs.append(p.p_value)
        p.extend([0.05] * 8)
        # 当前证据大幅回落，但 p 值不得变大。
        self.assertLessEqual(p.p_value, highs[0] + 1e-18)
        self.assertEqual(p.max_log_evidence, max(0.0, p.max_log_evidence))

    def test_p_value_capped_at_one_initially(self) -> None:
        p = MeanEvidenceProcess(M, GRID, ALPHA)
        # 证据从不上 1 时 p 恒为 1（max E_s 含 E_0 = 1）。
        p.extend([0.5, 0.5, 0.49])
        self.assertEqual(p.p_value, 1.0)

    def test_analyze_helper(self) -> None:
        xs = [0.9] * 30
        r = analyze(M, GRID, xs, ALPHA)
        self.assertIsInstance(r, AnalysisResult)
        self.assertEqual(r.t, 30)
        self.assertEqual(r.alpha, ALPHA)
        self.assertIsNotNone(r.stop_time)
        self.assertTrue(r.rejected)
        self.assertLessEqual(r.p_value, ALPHA + 1e-15)

    def test_threshold_boundary_is_inclusive(self) -> None:
        # 精确二进制可表示的等号情形：m=0.5, 网格 [2.0]=1/m，单点 x=1，
        # W=2，E=2；alpha=0.5，阈值 1/alpha=2 恰被命中（>= 含等号）。
        p = MeanEvidenceProcess(0.5, [2.0], 0.5)
        p.append(1.0)
        self.assertEqual(p.evidence, 2.0)
        self.assertEqual(p.stop_time, 1)
        # 差一步不到阈值则不停止。
        q = MeanEvidenceProcess(0.5, [0.0, 2.0], 0.5)  # E = (1+2)/2 = 1.5 < 2
        q.append(1.0)
        self.assertIsNone(q.stop_time)

    def test_extend_atomicity_on_bad_batch(self) -> None:
        p = MeanEvidenceProcess(M, GRID, ALPHA)
        p.extend([0.9, 0.9])
        t_before, lw_before = p.t, p.log_wealth
        with self.assertRaises(InvalidObservationError):
            p.extend([0.9, 1.5, 0.9])  # 中间非法
        self.assertEqual(p.t, t_before)
        self.assertEqual(p.log_wealth, lw_before)
        with self.assertRaises(InvalidObservationError):
            p.extend([float("nan"), 0.9])
        self.assertEqual(p.t, t_before)

    def test_extend_atomicity_kept_even_when_first_bad(self) -> None:
        p = MeanEvidenceProcess(M, GRID, ALPHA)
        with self.assertRaises(InvalidObservationError):
            p.extend([-0.01])
        self.assertEqual(p.t, 0)
        self.assertEqual(p.evidence, 1.0)


class InvalidInputTests(unittest.TestCase):
    def test_bad_m(self) -> None:
        for bad in (0.0, 1.0, -0.1, 1.1):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidParameterError):
                    MeanEvidenceProcess(bad, [0.0], ALPHA)

    def test_m_rejects_nonfinite_and_bool(self) -> None:
        for bad in (float("nan"), float("inf"), float("-inf"), True, False):
            with self.subTest(bad=bad):
                with self.assertRaises((InvalidParameterError, TypeError)):
                    MeanEvidenceProcess(bad, [0.0], ALPHA)

    def test_illegal_lambda_outside_range(self) -> None:
        # 0 <= lambda <= 1/m
        with self.assertRaises(InvalidParameterError):
            MeanEvidenceProcess(0.5, [2.0000001], ALPHA)
        with self.assertRaises(InvalidParameterError):
            MeanEvidenceProcess(0.5, [-1e-12], ALPHA)
        with self.assertRaises(InvalidParameterError):
            MeanEvidenceProcess(0.25, [5.0], ALPHA)  # 1/m = 4

    def test_grid_must_be_nonempty_finite(self) -> None:
        with self.assertRaises(InvalidParameterError):
            MeanEvidenceProcess(0.5, [], ALPHA)
        with self.assertRaises(InvalidParameterError):
            MeanEvidenceProcess(0.5, [0.0, float("nan")], ALPHA)
        with self.assertRaises(InvalidParameterError):
            MeanEvidenceProcess(0.5, 0.5, ALPHA)  # type: ignore[arg-type]

    def test_bad_alpha(self) -> None:
        for bad in (0.0, 1.0, -0.1, 2.0, float("nan"), True):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidParameterError):
                    MeanEvidenceProcess(M, [0.0], bad)

    def test_bad_observations(self) -> None:
        p = MeanEvidenceProcess(M, GRID, ALPHA)
        for bad in (-0.001, 1.001, float("nan"), float("inf"), float("-inf"),
                    True, False, "0.5", None):
            with self.subTest(bad=bad):
                with self.assertRaises((InvalidObservationError, InvalidParameterError, TypeError)):
                    p.append(bad)
                self.assertEqual(p.t, 0)

    def test_error_message_names_parameter(self) -> None:
        try:
            MeanEvidenceProcess(0.5, [3.0], ALPHA)
        except InvalidParameterError as exc:
            self.assertIn("lambda_grid[0]", str(exc))
        else:
            self.fail("应当抛 InvalidParameterError")
        try:
            MeanEvidenceProcess(float("nan"), [0.0], ALPHA)
        except InvalidParameterError as exc:
            self.assertIn("m", str(exc))
        else:
            self.fail("应当抛 InvalidParameterError")


if __name__ == "__main__":
    unittest.main()
