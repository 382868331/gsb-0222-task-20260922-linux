"""独立参考测试：精确 Fraction 枚举，不导入被测核心 mean_evidence。

期望值（含期望财富、命中概率、精确 E_t/p_t）全部由本文件内的独立
有理数实现给出；mean_evidence 仅在对照测试中被导入，且期望值不来自它。
"""

from __future__ import annotations

import math
import unittest
from fractions import Fraction

import mean_evidence as me


# ---------------------------------------------------------------------------
# 独立参考实现：全程 Fraction 精确有理运算
# ---------------------------------------------------------------------------

def _fr(x) -> Fraction:
    """把被测代码将看到的 float/int 输入按其精确值转为 Fraction。"""
    return Fraction(float(x))


class RefProcess:
    """与产品代码算法路径无关的朴素精确参考。"""

    def __init__(self, m, lambdas, alpha):
        self.m = _fr(m)
        self.lams = [_fr(v) for v in lambdas]
        self.alpha = _fr(alpha)
        self.wealth = [Fraction(1) for _ in self.lams]
        self.max_e = Fraction(1)
        self.t = 0
        self.stop_time = None

    def update(self, x):
        fx = _fr(x)
        for i, lam in enumerate(self.lams):
            self.wealth[i] *= Fraction(1) + lam * (fx - self.m)
        self.t += 1
        e = sum(self.wealth, Fraction(0)) / len(self.lams)
        if e > self.max_e:
            self.max_e = e
        if self.stop_time is None and e >= Fraction(1) / self.alpha:
            self.stop_time = self.t
        return e


def bernoulli_path_prob(path: tuple[int, ...], p: Fraction) -> Fraction:
    q = Fraction(1) - p
    prob = Fraction(1)
    for bit in path:
        prob *= p if bit == 1 else q
    return prob


def expected_wealth(m: Fraction, p: Fraction, lam: Fraction, n: int) -> Fraction:
    """E0[W_lambda(n)]：2^n 条 Bernoulli 路径精确枚举（叶子带路径概率）。"""
    total = Fraction(0)

    def walk(t, w, prob):
        nonlocal total
        if t == n:
            total += prob * w
            return
        walk(t + 1, w * (1 + lam * (1 - m)), prob * p)      # x=1
        walk(t + 1, w * (1 + lam * (-m)), prob * (1 - p))  # x=0

    walk(0, Fraction(1), Fraction(1))
    return total


def hit_mass(m: Fraction, p: Fraction, lams: list[Fraction], alpha: Fraction,
             n: int) -> Fraction:
    """P(存在 t≤n 使 E_t ≥ 1/alpha)：按首次命中路径精确枚举。"""
    threshold = Fraction(1) / alpha
    k = len(lams)
    hit = Fraction(0)

    def walk(t, wealth, prob, already_hit):
        nonlocal hit
        if t == n:
            if already_hit:
                hit += prob
            return
        if already_hit:
            # 首次命中后所有延续路径都计入命中质量
            walk(t + 1, wealth, prob * p, True)
            walk(t + 1, wealth, prob * (1 - p), True)
            return
        w1 = [w * (1 + lams[i] * (1 - m)) for i, w in enumerate(wealth)]
        w0 = [w * (1 + lams[i] * (-m)) for i, w in enumerate(wealth)]
        walk(t + 1, w1, prob * p, sum(w1) / k >= threshold)
        walk(t + 1, w0, prob * (1 - p), sum(w0) / k >= threshold)

    walk(0, [Fraction(1)] * k, Fraction(1), False)
    return hit


# ---------------------------------------------------------------------------
# 确定性数据
# ---------------------------------------------------------------------------

def lcg_values(n: int, seed: int = 20260923) -> list[float]:
    """仓库内确定性 LCG，无外部依赖。"""
    out = []
    state = seed
    for _ in range(n):
        state = (1103515245 * state + 12345) % (2 ** 31)
        u = state / (2 ** 31)
        out.append(0.1 + 0.6 * u)  # 落在 [0.1, 0.7)
    return out


class IndependentReferenceTests(unittest.TestCase):
    # -- 两点 Bernoulli 短路径枚举：H0（p ≤ m）下 E[W] ≤ 1 -----------------

    def test_expected_wealth_le_one_h0(self):
        cases = [
            (m, p, lam, n)
            for m in (Fraction(1, 2), Fraction(1, 4), Fraction(3, 10))
            for p in (Fraction(1, 2), Fraction(1, 4), Fraction(1, 10))
            if p <= m
            for lam in (Fraction(0), Fraction(1) / m, Fraction(1) / (2 * m))
            for n in (1, 2, 3, 5)
        ]
        for m, p, lam, n in cases:
            with self.subTest(m=m, p=p, lam=lam, n=n):
                ew = expected_wealth(m, p, lam, n)
                self.assertLessEqual(ew, Fraction(1))

    def test_lambda_star_expected_wealth_equality(self):
        # λ*=1/m、p=m 时各因子条件期望恰为 1，E[W] 恒为 1（精确等于）。
        for m in (Fraction(1, 2), Fraction(1, 4), Fraction(2, 5)):
            for n in (1, 3, 6):
                ew = expected_wealth(m, m, Fraction(1) / m, n)
                self.assertEqual(ew, Fraction(1))

    def test_optional_stopping_hit_mass_le_alpha(self):
        # 独立枚举 Ville 不等式：H0=p=m=1/2，单轨道 λ*=2，阈值 4，
        # 命中要求前两个观测均为 1，路径质量恰为 1/4 ≤ α=1/4（界紧）。
        alpha = Fraction(1, 4)
        m = Fraction(1, 2)
        mass = hit_mass(m, m, [Fraction(2)], alpha, n=4)
        self.assertEqual(mass, alpha)

    def test_hit_mass_boundary_equals_alpha(self):
        # n=3 同样枚举：阈值 4 在 t=2 由路径前缀 11 首次命中，质量 1/4=α。
        alpha = Fraction(1, 4)
        m = Fraction(1, 2)
        mass = hit_mass(m, m, [Fraction(2)], alpha, n=3)
        self.assertEqual(mass, alpha)

    # -- 产品代码 vs 独立精确参考（期望值不来自被测代码） -------------------

    def test_matches_fraction_oracle_random_stream(self):
        m = 0.3
        grid = [0.0, 1.0, 2.0, 1.0 / m]
        alpha = 0.2
        xs = lcg_values(40)

        ref = RefProcess(m, grid, alpha)
        proc = me.MeanEvidenceProcess(m, grid, alpha)
        for x in xs:
            ref.update(x)
            snap = proc.update(x)
            e_ref = float(sum(ref.wealth) / len(ref.wealth))
            # 独立基准：log(精确混合 E)；只允许舍入级误差，不放宽结构边界。
            log_ref = math.log(e_ref)
            scale = max(1.0, abs(log_ref))
            self.assertLess(
                abs(snap.log_evidence - log_ref), 1e-12 * scale,
                msg=f"t={ref.t} core={snap.log_evidence} ref={log_ref}",
            )
            self.assertAlmostEqual(
                snap.max_log_evidence, math.log(float(ref.max_e)),
                delta=1e-12 * max(1.0, abs(math.log(float(ref.max_e)))),
            )
            self.assertEqual(snap.stop_time, ref.stop_time)
            p_ref = min(1.0, 1.0 / float(ref.max_e))
            self.assertAlmostEqual(snap.p_value, p_ref, delta=1e-12)

    def test_p_value_and_stop_against_exact_fractions(self):
        m = 0.5
        grid = [0.0, 2.0]
        alpha = 0.5  # 阈值 2
        proc = me.MeanEvidenceProcess(m, grid, alpha)
        ref = RefProcess(m, grid, alpha)
        stream = [1, 1, 0, 0, 1]
        # t=1: 混合 E=(1+2)/2=1.5 < 2；t=2: E=(1+4)/2=2.5 ≥ 2 → 首次 t=2
        expected_stop = 2
        for i, x in enumerate(stream):
            ref.update(x)
            snap = proc.update(x)
        self.assertEqual(snap.stop_time, expected_stop)
        self.assertEqual(snap.stop_time, ref.stop_time)
        # 精确财富：λ=2 轨道 2*2*0...=0（t=3 归零），λ=0 恒 1
        # E_5 = 1/2，max E=2.5（t=2），p=0.4
        self.assertEqual(proc.log_wealth()[0], 0.0)
        self.assertEqual(proc.log_wealth()[1], -math.inf)
        self.assertAlmostEqual(snap.evidence, 0.5, places=15)
        self.assertAlmostEqual(snap.max_evidence, 2.5, places=12)
        self.assertAlmostEqual(snap.p_value, 0.4, places=12)


if __name__ == "__main__":
    unittest.main()
