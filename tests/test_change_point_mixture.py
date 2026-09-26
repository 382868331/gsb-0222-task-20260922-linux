"""固定起点权重混合的独立参考测试。"""

from __future__ import annotations

import unittest
from fractions import Fraction

import mean_evidence as me


def fraction_value(x) -> Fraction:
    return Fraction(float(x))


class FractionSuffixProcess:
    def __init__(self, m, lambdas):
        self.m = Fraction(float(m))
        self.lambdas = [Fraction(float(v)) for v in lambdas]

    def evidence(self, xs: list[float]) -> Fraction:
        total = Fraction(0)
        for lam in self.lambdas:
            wealth = Fraction(1)
            for x in xs:
                wealth *= Fraction(1) + lam * (Fraction(float(x)) - self.m)
            total += wealth
        return total / len(self.lambdas)


class FractionSuffixMixture:
    def __init__(self, m, lambdas, alpha, weights, max_starts):
        self.ref = FractionSuffixProcess(m, lambdas)
        self.alpha = Fraction(float(alpha))
        self.weights = [Fraction(float(w)) for w in weights]
        self.max_starts = max_starts
        self.xs: list[float] = []
        self.max_e = Fraction(1)
        self.stop_time = None

    def update(self, x):
        self.xs.append(x)
        evidence = self.evidence()
        if evidence > self.max_e:
            self.max_e = evidence
        if self.stop_time is None and evidence >= Fraction(1) / self.alpha:
            self.stop_time = len(self.xs)
        return evidence

    def component_evidence(self):
        values = []
        for s in range(self.max_starts):
            if s < len(self.xs):
                values.append(self.ref.evidence(self.xs[s:]))
            else:
                values.append(Fraction(1))
        return values

    def evidence(self):
        return sum(
            weight * value
            for weight, value in zip(self.weights, self.component_evidence())
        )


class ChangePointMixtureTests(unittest.TestCase):
    def test_initial_and_future_start_mass_is_retained(self):
        proc = me.ChangePointEvidenceProcess(
            0.5, [0.0, 2.0], 0.2, [0.75, 0.25], 2
        )
        snap = proc.snapshot()
        self.assertEqual(snap.t, 0)
        self.assertEqual(snap.evidence, 1.0)
        self.assertEqual(snap.component_evidence, (1.0, 1.0))
        self.assertEqual(snap.contributions, (0.75, 0.25))

        snap = proc.update(1)
        self.assertEqual(snap.active_starts, 1)
        self.assertEqual(snap.component_evidence, (1.5, 1.0))
        self.assertEqual(snap.contributions, (1.125, 0.25))
        self.assertAlmostEqual(snap.evidence, 1.375)

    def test_each_suffix_is_restarted_independently_late_change(self):
        proc = me.ChangePointEvidenceProcess(
            0.5, [0.0, 2.0], 0.2, [0.25, 0.75], 2
        )
        ref = FractionSuffixMixture(
            0.5, [0.0, 2.0], 0.2, [0.25, 0.75], 2
        )
        for x in (0, 1):
            snap = proc.update(x)
            expected = ref.update(x)
            self.assertAlmostEqual(snap.evidence, float(expected), places=14)

        self.assertEqual(snap.active_starts, 2)
        self.assertEqual(snap.component_evidence, (0.5, 1.5))
        self.assertEqual(snap.contributions, (0.125, 1.125))
        self.assertAlmostEqual(snap.evidence, 1.25)

    def test_no_change_keeps_mixture_at_one(self):
        proc = me.ChangePointEvidenceProcess(
            0.5, [0.0, 2.0], 0.1, [0.1, 0.2, 0.3, 0.4], 4
        )
        ref = FractionSuffixMixture(
            0.5, [0.0, 2.0], 0.1, [0.1, 0.2, 0.3, 0.4], 4
        )
        for t, x in enumerate([0.5] * 4, start=1):
            snap = proc.update(x)
            expected = ref.update(x)
            self.assertEqual(snap.evidence, float(expected))
            self.assertEqual(snap.evidence, 1.0)
            self.assertEqual(snap.active_starts, t)
            self.assertIsNone(snap.stop_time)

    def test_zero_weight_component_does_not_amplify_early_evidence(self):
        proc = me.ChangePointEvidenceProcess(
            0.5, [0.0, 2.0], 0.2, [0.0, 1.0], 2
        )
        snap = proc.update(1)
        self.assertEqual(snap.component_evidence, (1.5, 1.0))
        self.assertEqual(snap.contributions, (0.0, 1.0))
        self.assertEqual(snap.evidence, 1.0)

        snap = proc.update(1)
        self.assertEqual(snap.component_evidence, (2.5, 1.5))
        self.assertEqual(snap.contributions, (0.0, 1.5))
        self.assertAlmostEqual(snap.evidence, 1.5)

    def test_first_alarm_freezes_but_updates_continue_by_original_rule(self):
        proc = me.ChangePointEvidenceProcess(
            0.5, [0.0, 2.0], 0.5, [0.5, 0.5], 2
        )
        first_two = [proc.update(x) for x in (1, 1)]
        self.assertIsNone(first_two[0].stop_time)
        self.assertEqual(first_two[1].stop_time, 2)
        self.assertAlmostEqual(first_two[1].evidence, 2.0)

        snap = proc.update(0)
        self.assertEqual(snap.stop_time, 2)
        self.assertEqual(snap.t, 3)
        self.assertEqual(snap.component_evidence, (0.5, 0.5))
        self.assertEqual(snap.contributions, (0.25, 0.25))
        self.assertAlmostEqual(snap.evidence, 0.5)

    def test_max_starts_limits_opened_components(self):
        proc = me.ChangePointEvidenceProcess(
            0.5, [0.0, 2.0], 0.2, [0.5, 0.5], 2
        )
        proc.extend([1, 0, 1, 0])
        snap = proc.snapshot()
        self.assertEqual((snap.t, snap.active_starts), (4, 2))
        # 只允许前两个起点；两个已开启分量仍按原规则处理后续输入。
        self.assertEqual(snap.component_evidence, (0.5, 0.5))
        self.assertEqual(snap.contributions, (0.25, 0.25))

    def test_matches_independent_suffix_oracle_on_short_stream(self):
        weights = [0.1, 0.2, 0.3, 0.4]
        stream = [1, 0, 1, 1]
        proc = me.ChangePointEvidenceProcess(
            0.5, [0.0, 1.0, 2.0], 0.1, weights, 4
        )
        ref = FractionSuffixMixture(
            0.5, [0.0, 1.0, 2.0], 0.1, weights, 4
        )
        for x in stream:
            snap = proc.update(x)
            expected = ref.update(x)
            self.assertAlmostEqual(snap.evidence, float(expected), places=12)
            self.assertEqual(snap.stop_time, ref.stop_time)
            for got, exp in zip(snap.component_evidence, ref.component_evidence()):
                self.assertAlmostEqual(got, float(exp), places=12)

    def test_invalid_weights_and_max_starts(self):
        with self.assertRaises(ValueError):
            me.ChangePointEvidenceProcess(0.5, [0.0], 0.2, [0.5, 0.4], 2)
        with self.assertRaises(ValueError):
            me.ChangePointEvidenceProcess(0.5, [0.0], 0.2, [-0.1, 1.1], 2)
        with self.assertRaises(ValueError):
            me.ChangePointEvidenceProcess(0.5, [0.0], 0.2, [1.0], 2)
        with self.assertRaises(ValueError):
            me.ChangePointEvidenceProcess(0.5, [0.0], 0.2, [], 0)
        with self.assertRaises(TypeError):
            me.ChangePointEvidenceProcess(0.5, [0.0], 0.2, [0.0, 1.0], True)
        with self.assertRaises(TypeError):
            me.ChangePointEvidenceProcess(0.5, [0.0], 0.2, "01", 2)

    def test_extend_is_atomic(self):
        proc = me.ChangePointEvidenceProcess(
            0.5, [0.0, 2.0], 0.2, [0.5, 0.5], 2
        )
        proc.update(1)
        before = proc.snapshot()
        with self.assertRaises(ValueError):
            proc.extend([0, 1.5])
        self.assertEqual(proc.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
