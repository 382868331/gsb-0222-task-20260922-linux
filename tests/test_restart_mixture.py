"""重启起点混合检测器的独立参考与契约测试。

精确参考在本文件内对每个已开启后缀重新枚举 λ 财富，不调用被测检测器
产生期望值。
"""

from __future__ import annotations

import math
import unittest
from fractions import Fraction

from mean_evidence import RestartEvidenceMixture


def fraction_float(value) -> Fraction:
    return Fraction(float(value))


class ExactRestartReference:
    def __init__(self, m, lambdas, alpha, weights):
        self.m = fraction_float(m)
        self.lambdas = [fraction_float(v) for v in lambdas]
        self.alpha = fraction_float(alpha)
        self.weights = [fraction_float(v) for v in weights]
        self.xs: list[Fraction] = []
        self.max_e = Fraction(1)
        self.stop_time = None

    def component_evidence(self, zero_based_start: int) -> Fraction:
        values = self.xs[zero_based_start:]
        total = Fraction(0)
        for lam in self.lambdas:
            wealth = Fraction(1)
            for x in values:
                wealth *= Fraction(1) + lam * (x - self.m)
            total += wealth
        return total / len(self.lambdas)

    def update(self, x):
        self.xs.append(fraction_float(x))
        t = len(self.xs)
        total = Fraction(0)
        contributions = []
        for start in range(min(t, len(self.weights))):
            contribution = self.weights[start] * self.component_evidence(start)
            contributions.append(contribution)
            total += contribution
        future = sum(self.weights[t:], Fraction(0))
        total += future
        if total > self.max_e:
            self.max_e = total
        if self.stop_time is None and total >= Fraction(1) / self.alpha:
            self.stop_time = t
        return total, contributions, future


class RestartMixtureIndependentReferenceTests(unittest.TestCase):
    def assert_matches_reference(self, stream, weights, m=0.5,
                                 lambdas=(0.0, 2.0), alpha=0.2):
        proc = RestartEvidenceMixture(m, list(lambdas), alpha, weights)
        ref = ExactRestartReference(m, lambdas, alpha, weights)

        for t, x in enumerate(stream, start=1):
            expected, expected_contrib, expected_future = ref.update(x)
            snap = proc.update(x)

            self.assertEqual(snap.t, t)
            self.assertEqual(proc.active_start_count(), min(t, len(weights)))
            self.assertAlmostEqual(
                snap.log_evidence, math.log(float(expected)),
                delta=1e-12 * max(1.0, abs(math.log(float(expected)))),
            )
            self.assertAlmostEqual(snap.evidence, float(expected), delta=1e-11)
            self.assertAlmostEqual(
                snap.future_weight, float(expected_future), places=14
            )
            for i, value in enumerate(expected_contrib):
                self.assertAlmostEqual(
                    snap.start_contributions[i], float(value), places=12
                )
            self.assertEqual(len(snap.start_contributions), len(weights))
            self.assertEqual(
                snap.start_contributions[len(expected_contrib):],
                (0.0,) * (len(weights) - len(expected_contrib)),
            )
            self.assertEqual(snap.stop_time, ref.stop_time)

    def test_no_change_zero_stream_keeps_future_mass(self):
        weights = [0.25, 0.25, 0.25, 0.25]
        self.assert_matches_reference([0, 0, 0], weights)
        proc = RestartEvidenceMixture(0.5, [0.0, 2.0], 0.1, weights)
        snap = proc.extend([0, 0, 0])
        # 三个已开启分量各为 0.25*0.5，尚未到来的质量仍按 1 保留。
        self.assertAlmostEqual(snap.evidence, 0.625)
        self.assertAlmostEqual(snap.future_weight, 0.25)
        self.assertIsNone(snap.stop_time)

    def test_late_change_is_measured_from_each_suffix(self):
        weights = [0.25, 0.25, 0.25, 0.25]
        # 前三点为 0，之后连续为 1；最晚分量没有早期损失。
        self.assert_matches_reference([0, 0, 0, 1, 1, 1], weights)
        proc = RestartEvidenceMixture(0.5, [0.0, 2.0], 0.1, weights)
        snap = proc.extend([0, 0, 0, 1, 1, 1])
        self.assertGreater(snap.start_contributions[3], snap.start_contributions[0])
        self.assertGreater(snap.evidence, 1.0)

    def test_zero_weight_component_updates_but_contributes_nothing(self):
        weights = [1.0, 0.0]
        proc = RestartEvidenceMixture(0.5, [0.0, 2.0], 2.0 / 3.0, weights)
        snap = proc.extend([1, 1])
        self.assertEqual(proc.active_start_count(), 2)
        self.assertEqual(snap.start_contributions[1], 0.0)
        self.assertEqual(snap.log_start_contributions[1], -math.inf)
        self.assertGreater(proc.component_snapshot(2).evidence, 1.0)
        self.assertAlmostEqual(snap.evidence, snap.start_contributions[0])

    def test_alarm_then_continuation_matches_suffix_reference(self):
        weights = [1.0, 0.0]
        self.assert_matches_reference(
            [1, 1, 0, 0], weights, alpha=0.5
        )


class RestartMixtureContractTests(unittest.TestCase):
    def test_initial_mass_is_one_without_opening_component(self):
        proc = RestartEvidenceMixture(0.5, [0.0, 2.0], 0.1, [0.3, 0.7])
        snap = proc.snapshot()
        self.assertEqual((snap.t, snap.evidence, snap.future_weight), (0, 1.0, 1.0))
        self.assertEqual(snap.start_contributions, (0.0, 0.0))
        self.assertIsNone(snap.stop_time)

    def test_unopened_mass_not_renormalized_into_early_evidence(self):
        proc = RestartEvidenceMixture(
            0.5, [0.0, 2.0], 0.1, [0.25, 0.75], max_starts=2
        )
        snap = proc.update(1)
        # 0.25 * 1.5 + 0.75，而不是把 0.25 归一化成 1。
        self.assertAlmostEqual(snap.evidence, 1.125)
        self.assertAlmostEqual(snap.start_contributions[0], 0.375)
        self.assertAlmostEqual(snap.future_weight, 0.75)

    def test_after_max_starts_no_more_components_or_future_mass(self):
        proc = RestartEvidenceMixture(0.5, [0.0, 2.0], 0.1, [0.4, 0.6])
        proc.extend([1, 1, 0])
        self.assertEqual(proc.active_start_count(), 2)
        self.assertEqual(proc.t, 3)
        self.assertEqual(proc.snapshot().future_weight, 0.0)

    def test_alarm_freezes_first_time_but_values_continue(self):
        proc = RestartEvidenceMixture(0.5, [0.0, 2.0], 2.0 / 3.0, [1.0, 0.0])
        first = proc.update(1)
        self.assertEqual(first.stop_time, 1)
        later = proc.extend([0, 0])
        self.assertEqual(later.stop_time, 1)
        self.assertEqual(later.t, 3)
        self.assertAlmostEqual(later.evidence, 0.5)
        self.assertAlmostEqual(later.max_evidence, 1.5)

    def test_invalid_weights_and_max_starts(self):
        with self.assertRaisesRegex(ValueError, "总和"):
            RestartEvidenceMixture(0.5, [0.0], 0.1, [0.5, 0.25])
        with self.assertRaisesRegex(ValueError, "非负"):
            RestartEvidenceMixture(0.5, [0.0], 0.1, [-0.1, 1.1])
        with self.assertRaises(ValueError):
            RestartEvidenceMixture(0.5, [0.0], 0.1, [0.5, 0.5], max_starts=3)
        with self.assertRaises(TypeError):
            RestartEvidenceMixture(0.5, [0.0], 0.1, [0.5, 0.5], max_starts=True)
        with self.assertRaisesRegex(ValueError, "正 int"):
            RestartEvidenceMixture(0.5, [0.0], 0.1, [0.5, 0.5], max_starts=0)

    def test_extend_is_atomic(self):
        proc = RestartEvidenceMixture(0.5, [0.0, 2.0], 0.1, [0.5, 0.5])
        proc.update(1)
        before = (proc.t, proc.active_start_count(), proc.snapshot())
        with self.assertRaises(ValueError):
            proc.extend([1, 1.5])
        after = (proc.t, proc.active_start_count(), proc.snapshot())
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
