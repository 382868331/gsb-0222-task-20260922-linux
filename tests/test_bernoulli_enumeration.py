"""Bernoulli 两点分布短路径全枚举（独立精确参考）。

不调用被测核心，只用 ``fractions.Fraction`` 对长度 n 的全部 2^n 条
路径精确枚举，验证：

1. 零假设（Bernoulli 参数 p <= m）下单个财富 W_lambda 的期望 <= 1；
2. 均匀混合证据 E_t 的期望 <= 1；
3. 可选停止（在 E_t 首次 >= 1/alpha 时停止，否则走到 n）下，
   停止后财富的期望依然 <= 1（Ville 不等式的有限情形）；
4. 包含 lambda = 1/m 且观测取 0 的零因子路径。
"""

from __future__ import annotations

import unittest
from fractions import Fraction
from itertools import product

from mean_evidence import MeanEvidenceProcess


def _stopped_wealth_expectation(
    m: Fraction, p: Fraction, grid: list[Fraction], n: int, alpha: Fraction
) -> Fraction:
    """枚举全部路径，独立模拟“首次 E_t >= 1/alpha 即停”的期望终端财富。

    停止路径只按其停时前缀长度计概率（未停止路径长度为 n）。
    """
    threshold = 1 / alpha
    total = Fraction(0)
    for path in product((0, 1), repeat=n):
        wealth = [Fraction(1) for _ in grid]
        terminal = None
        for x in path:
            for k, lam in enumerate(grid):
                wealth[k] *= 1 + lam * (Fraction(x) - m)
            evidence = sum(wealth, Fraction(0)) / len(grid)
            if terminal is None and evidence >= threshold:
                terminal = evidence  # 冻结终端值，仍继续扫描以枚举全部全长路径
        if terminal is None:
            terminal = sum(wealth, Fraction(0)) / len(grid)
        # 每条全长路径按全长概率计；同一停止前缀的所有连续路径概率
        # 之和恰为该前缀的概率。
        prob = p ** sum(path) * (1 - p) ** (n - sum(path))
        total += terminal * prob
    return total


def _terminal_wealth_expectation(
    m: Fraction, p: Fraction, grid: list[Fraction], n: int
) -> tuple[Fraction, list[Fraction]]:
    """返回 (E[E_n], [E[W_lambda,n] ...])，精确分数。"""
    e_single = [Fraction(0) for _ in grid]
    for path in product((0, 1), repeat=n):
        prob = p ** sum(path) * (1 - p) ** (n - sum(path))
        for k, lam in enumerate(grid):
            w = Fraction(1)
            for x in path:
                w *= 1 + lam * (Fraction(x) - m)
            e_single[k] += w * prob
    # 期望与均匀混合可交换：E[E_n] = (1/K) Σ_k E[W_k,n]。
    e_mix = sum(e_single, Fraction(0)) / len(grid)
    return e_mix, e_single


class BernoulliEnumerationTests(unittest.TestCase):
    def test_expected_wealth_le_one_under_null(self) -> None:
        # (m, p, n)：p <= m 的若干零假设情形。
        cases = [
            (Fraction(1, 2), Fraction(0), 6),
            (Fraction(1, 2), Fraction(1, 2), 6),
            (Fraction(1, 3), Fraction(1, 4), 5),
            (Fraction(3, 4), Fraction(1, 2), 5),
            (Fraction(7, 10), Fraction(7, 10), 6),
        ]
        for m, p, n in cases:
            with self.subTest(m=m, p=p, n=n):
                grid = [Fraction(0), Fraction(1, 2) / m, Fraction(1) / m]
                e_mix, e_single = _terminal_wealth_expectation(m, p, grid, n)
                self.assertLessEqual(e_mix, Fraction(1))
                for k, v in enumerate(e_single):
                    self.assertLessEqual(v, Fraction(1), f"grid[{k}]")

    def test_zero_factor_paths_included(self) -> None:
        # lambda = 1/m：x=0 时因子恰为 0，财富永久归零；
        # p=0 的零假设下，全部路径所有因子非负且期望必须 <= 1。
        m = Fraction(1, 2)
        grid = [Fraction(2)]  # 1/m
        e_mix, e_single = _terminal_wealth_expectation(m, Fraction(0), grid, 4)
        self.assertEqual(e_single[0], Fraction(0))  # 只有全 0 路径：财富为 0
        self.assertLessEqual(e_mix, Fraction(1))

    def test_optional_stopping_expected_wealth_le_one(self) -> None:
        m = Fraction(1, 2)
        grid = [Fraction(0), Fraction(1), Fraction(2)]
        alpha = Fraction(1, 2)  # 阈值 2
        n = 6
        for p in (Fraction(0), Fraction(1, 4), Fraction(1, 2)):
            with self.subTest(p=p):
                v = _stopped_wealth_expectation(m, p, grid, n, alpha)
                self.assertLessEqual(v, Fraction(1))
                self.assertGreaterEqual(v, Fraction(0))

    def test_stopping_rule_matches_core_on_enumerated_paths(self) -> None:
        # 核心过程与精确枚举在同一条路径上的停止时刻必须一致
        # （停止语义的一致性核对，期望值本身仍由上面的独立枚举给出）。
        m = 0.5
        grid = [0.0, 1.0, 2.0]
        alpha = 0.5
        threshold = Fraction(2)
        for path in product((0, 1), repeat=7):
            proc = MeanEvidenceProcess(m, grid, alpha)
            core_stop = None
            for t, x in enumerate(path, start=1):
                proc.append(x)
                if proc.stop_time is not None:
                    core_stop = t
                    break
            # 独立枚举参考。
            wealth = [Fraction(1) for _ in grid]
            ref_stop = None
            for t, x in enumerate(path, start=1):
                for k, lam in enumerate(grid):
                    wealth[k] *= 1 + Fraction(lam) * (Fraction(x) - Fraction(m))
                evidence = sum(wealth, Fraction(0)) / len(grid)
                if evidence >= threshold:
                    ref_stop = t
                    break
            self.assertEqual(core_stop, ref_stop, path)


if __name__ == "__main__":
    unittest.main()
