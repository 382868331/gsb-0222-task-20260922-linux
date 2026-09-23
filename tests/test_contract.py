"""契约与边界测试：0/1 边界、零因子、状态恢复、非法 λ、错误语义、原子性。"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import unittest

import mean_evidence as me
from mean_evidence import EvidenceSnapshot, MeanEvidenceProcess


class SnapshotBasicsTests(unittest.TestCase):
    def test_initial_state_t0(self):
        p = MeanEvidenceProcess(0.5, [0.0, 2.0], 0.05)
        s = p.snapshot()
        self.assertEqual(s.t, 0)
        self.assertEqual(s.evidence, 1.0)
        self.assertEqual(s.log_evidence, 0.0)
        self.assertEqual(s.max_evidence, 1.0)
        self.assertEqual(s.max_log_evidence, 0.0)
        self.assertEqual(s.p_value, 1.0)
        self.assertAlmostEqual(s.threshold, 20.0)
        self.assertFalse(s.stopped)
        self.assertIsNone(s.stop_time)

    def test_snapshot_is_frozen(self):
        p = MeanEvidenceProcess(0.5, [0.0, 2.0], 0.5)
        s = p.update(1)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            s.t = 5  # type: ignore[misc]

    def test_duplicate_lambdas_counted(self):
        p = MeanEvidenceProcess(0.5, [2.0, 2.0], 0.5)
        s = p.update(1)  # 两条轨道 W 均为 2 → E=2
        self.assertAlmostEqual(s.evidence, 2.0)
        self.assertEqual(s.stop_time, 1)


class BoundaryInputsTests(unittest.TestCase):
    def test_x_zero_and_one_accepted(self):
        p = MeanEvidenceProcess(0.5, [0.0, 2.0], 0.5)
        s0 = p.update(0)
        self.assertEqual(s0.evidence, 0.5)
        s1 = p.update(1)
        self.assertAlmostEqual(s1.evidence, 0.5)  # λ=2 轨道上一步已归零

    def test_x_int_boundaries_accepted(self):
        p = MeanEvidenceProcess(0.5, [1.0], 0.5)
        self.assertEqual(p.update(0).evidence, 0.5)
        p = MeanEvidenceProcess(0.5, [1.0], 0.5)
        self.assertAlmostEqual(p.update(1).evidence, 1.5)

    def test_x_bool_rejected_even_though_numeric(self):
        p = MeanEvidenceProcess(0.5, [1.0], 0.5)
        with self.assertRaises(TypeError):
            p.update(True)
        self.assertEqual(p.t, 0)  # 未推进

    def test_m_boundaries_rejected(self):
        for bad in (0, 1, 0.0, 1.0, -0.1, 1.0 + 1e-12):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    MeanEvidenceProcess(bad, [0.5], 0.05)
        with self.assertRaises(TypeError):
            MeanEvidenceProcess(True, [0.5], 0.05)

    def test_alpha_boundaries_rejected(self):
        for bad in (0, 1, 0.0, 1.0, -0.5, 2.0):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    MeanEvidenceProcess(0.5, [1.0], bad)

    def test_nan_inf_rejected_everywhere(self):
        bads = (float("nan"), float("inf"), float("-inf"))
        for bad in bads:
            with self.subTest(bad=bad, where="m"):
                with self.assertRaises(ValueError):
                    MeanEvidenceProcess(bad, [1.0], 0.05)
            with self.subTest(bad=bad, where="alpha"):
                with self.assertRaises(ValueError):
                    MeanEvidenceProcess(0.5, [1.0], bad)
            with self.subTest(bad=bad, where="lambda"):
                with self.assertRaises(ValueError):
                    MeanEvidenceProcess(0.5, [bad], 0.05)
            with self.subTest(bad=bad, where="x"):
                p = MeanEvidenceProcess(0.5, [1.0], 0.05)
                with self.assertRaises(ValueError):
                    p.update(bad)

    def test_non_numeric_types_rejected(self):
        with self.assertRaises(TypeError):
            MeanEvidenceProcess("0.5", [1.0], 0.05)
        with self.assertRaises(TypeError):
            MeanEvidenceProcess(0.5, "01", 0.05)
        with self.assertRaises(TypeError):
            MeanEvidenceProcess(0.5, 1.0, 0.05)  # 非序列
        with self.assertRaises(TypeError):
            MeanEvidenceProcess(0.5, [1.0], 0.05).update(None)

    def test_empty_grid_rejected(self):
        with self.assertRaises(ValueError):
            MeanEvidenceProcess(0.5, [], 0.05)


class LambdaGridTests(unittest.TestCase):
    def test_lambda_zero_and_max_allowed(self):
        for m in (0.1, 0.5, 0.9):
            p = MeanEvidenceProcess(m, [0.0, 1.0 / m], 0.5)
            self.assertEqual(len(p.lambdas), 2)

    def test_lambda_negative_rejected(self):
        with self.assertRaises(ValueError):
            MeanEvidenceProcess(0.5, [-1e-16], 0.5)

    def test_lambda_above_one_over_m_rejected(self):
        m = 0.3
        with self.assertRaises(ValueError):
            MeanEvidenceProcess(m, [1.0 / m + 1e-9], 0.5)

    def test_lambda_nan_inf_bool_rejected(self):
        for bad in (float("nan"), float("inf"), True, False):
            with self.subTest(bad=bad):
                with self.assertRaises((ValueError, TypeError)):
                    MeanEvidenceProcess(0.5, [bad], 0.5)

    def test_lambda_max_zero_factor_exact(self):
        # m=0.1, λ*=10=1/m，x=0 时真实因子恰为 0（fma 单次舍入）。
        p = MeanEvidenceProcess(0.1, [10.0], 0.5)
        p.update(0)
        self.assertEqual(p.log_wealth(), (-math.inf,))

    def test_grid_immutable_from_outside(self):
        grid = [0.0, 1.0]
        p = MeanEvidenceProcess(0.5, grid, 0.5)
        grid.append(9.0)
        self.assertEqual(len(p.lambdas), 2)


class ZeroFactorTests(unittest.TestCase):
    def test_zero_factor_is_permanent(self):
        p = MeanEvidenceProcess(0.5, [2.0], 0.5)
        p.update(0)  # 因子 1 + 2(0-0.5) = 0
        self.assertEqual(p.log_wealth(), (-math.inf,))
        for _ in range(3):
            p.update(1)  # 0 乘任何因子仍为 0，不能“复活”
        self.assertEqual(p.log_wealth(), (-math.inf,))

    def test_all_tracks_dead_gives_zero_evidence_with_history(self):
        p = MeanEvidenceProcess(0.5, [2.0], 0.25)  # 阈值 4：本用例不命中
        up = p.update(1)   # E=2，max=2
        self.assertEqual(up.evidence, 2.0)
        p.update(0)        # 唯一轨道因子为 0，归零
        s = p.snapshot()
        self.assertEqual(s.evidence, 0.0)
        self.assertEqual(s.log_evidence, -math.inf)
        self.assertEqual(s.max_evidence, 2.0)  # 历史峰值保留
        self.assertAlmostEqual(s.p_value, 0.5)
        self.assertFalse(s.stopped)
        self.assertIsNone(s.stop_time)

    def test_all_dead_before_ever_rising(self):
        p = MeanEvidenceProcess(0.5, [2.0], 0.25)
        p.update(0)
        s = p.snapshot()
        self.assertEqual(s.evidence, 0.0)
        self.assertEqual(s.max_evidence, 1.0)
        self.assertEqual(s.p_value, 1.0)
        self.assertIsNone(s.stop_time)


class StopAndPValueTests(unittest.TestCase):
    def test_exact_threshold_hit_counts_as_stop(self):
        # m=1/2, grid {0,2}, α=2/3 → 阈值 3/2；x1=1 时 E1=3/2 恰命中。
        p = MeanEvidenceProcess(0.5, [0.0, 2.0], 2.0 / 3.0)
        s = p.update(1)
        self.assertAlmostEqual(s.evidence, 1.5)
        self.assertAlmostEqual(s.threshold, 1.5)
        self.assertEqual(s.stop_time, 1)
        self.assertTrue(s.stopped)
        self.assertLessEqual(s.p_value, 2.0 / 3.0 + 1e-15)

    def test_first_crossing_time_and_running_max(self):
        p = MeanEvidenceProcess(0.5, [0.0, 2.0], 0.5)  # 阈值 2
        times = []
        for x in [1, 1, 0, 0, 1]:
            s = p.update(x)
            times.append((s.t, s.max_evidence, s.stop_time))
        # E: 1.5, 2.5(首次命中 t=2), 归零后 .5, .5, .5；max 保留历史峰值
        self.assertEqual([t[2] for t in times], [None, 2, 2, 2, 2])
        maxes = [t[1] for t in times]
        self.assertEqual(maxes, sorted(maxes))
        self.assertAlmostEqual(maxes[0], 1.5)
        self.assertAlmostEqual(maxes[-1], 2.5)

    def test_p_value_uses_running_max(self):
        p = MeanEvidenceProcess(0.3, [0.0, 1.0], 0.1)
        s1 = p.update(1)
        e1 = s1.evidence
        s2 = p.update(0)
        self.assertLess(s2.evidence, s2.max_evidence)
        self.assertAlmostEqual(s2.p_value, min(1.0, 1.0 / e1))

    def test_continues_after_stop_recording(self):
        # 停止是报告事件，不禁止继续追加；stop_time 保持首次时刻。
        p = MeanEvidenceProcess(0.5, [2.0], 0.5)
        p.update(1)
        self.assertEqual(p.snapshot().stop_time, 1)
        p.update(1)
        self.assertEqual(p.snapshot().stop_time, 1)
        self.assertEqual(p.t, 2)


class ExtendAtomicityTests(unittest.TestCase):
    def test_extend_batch_matches_sequence(self):
        a = MeanEvidenceProcess(0.3, [0.0, 1.0, 2.0], 0.1)
        a.update(0.9)
        a.extend([0.1, 0.7, 0.2])
        b = MeanEvidenceProcess(0.3, [0.0, 1.0, 2.0], 0.1)
        b.update(0.9)
        for x in (0.1, 0.7, 0.2):
            b.update(x)
        self.assertEqual(a.log_wealth(), b.log_wealth())
        self.assertEqual(a.snapshot(), b.snapshot())

    def test_extend_accepts_generator(self):
        p = MeanEvidenceProcess(0.5, [1.0], 0.5)
        p.extend(float(x) for x in [0, 1, 1])
        self.assertEqual(p.t, 3)

    def test_extend_failure_leaves_no_partial_change(self):
        p = MeanEvidenceProcess(0.5, [0.0, 2.0], 0.5)
        p.extend([1, 0])
        before = p.to_dict()
        # 批次中第 3 个元素非法（越界）
        with self.assertRaises(ValueError):
            p.extend([1, 1, 1.5])
        self.assertEqual(p.to_dict(), before)
        # NaN 混入批次
        with self.assertRaises(ValueError):
            p.extend([1, float("nan"), 0])
        self.assertEqual(p.to_dict(), before)
        # bool 混入批次（TypeError）也不留部分变更
        with self.assertRaises(TypeError):
            p.extend([1, True, 0])
        self.assertEqual(p.to_dict(), before)

    def test_extend_string_rejected(self):
        p = MeanEvidenceProcess(0.5, [1.0], 0.5)
        with self.assertRaises(TypeError):
            p.extend("01")
        self.assertEqual(p.t, 0)


class SerializationTests(unittest.TestCase):
    def _streamed(self):
        p = MeanEvidenceProcess(0.3, [0.0, 1.0, 1.0 / 0.3], 0.07)
        p.extend([0.9, 0.2, 0.7, 0.1, 1.0, 0.0, 0.55])
        return p

    def test_json_round_trip_state_and_continue(self):
        p = self._streamed()
        text = p.to_json()
        json.loads(text)  # 必须是合法 JSON
        r = MeanEvidenceProcess.from_json(
            text, m=0.3, lambdas=[0.0, 1.0, 1.0 / 0.3], alpha=0.07
        )
        self.assertEqual(r.snapshot(), p.snapshot())
        self.assertEqual(r.log_wealth(), p.log_wealth())
        tail = [0.8, 0.05, 0.6]
        p.extend(tail)
        r.extend(tail)
        self.assertEqual(r.log_wealth(), p.log_wealth())
        self.assertEqual(r.snapshot(), p.snapshot())

    def test_restore_without_assertions_uses_stored_config(self):
        p = self._streamed()
        r = MeanEvidenceProcess.from_dict(p.to_dict())
        self.assertEqual(r.m, p.m)
        self.assertEqual(r.lambdas, p.lambdas)
        self.assertEqual(r.alpha, p.alpha)
        self.assertEqual(r.snapshot(), p.snapshot())

    def test_restore_rejects_m_grid_alpha_change(self):
        p = self._streamed()
        d = p.to_dict()
        with self.assertRaises(ValueError):
            MeanEvidenceProcess.from_dict(d, m=0.31)
        with self.assertRaises(ValueError):
            MeanEvidenceProcess.from_dict(d, lambdas=[0.0, 1.0])
        with self.assertRaises(ValueError):
            MeanEvidenceProcess.from_dict(d, lambdas=[0.0, 1.0, 1.0 / 0.3, 0.5])
        with self.assertRaises(ValueError):
            MeanEvidenceProcess.from_dict(d, alpha=0.05)
        # 断言值与状态一致时恢复成功
        MeanEvidenceProcess.from_dict(
            d, m=0.3, lambdas=[0.0, 1.0, 1.0 / 0.3], alpha=0.07
        )

    def test_restore_rejects_corrupted_states(self):
        d = self._streamed().to_dict()
        fresh = MeanEvidenceProcess(0.3, [0.0, 1.0, 1.0 / 0.3], 0.07).to_dict()

        bad_version = copy.deepcopy(d)
        bad_version["version"] = 99
        with self.assertRaises(ValueError):
            MeanEvidenceProcess.from_dict(bad_version)

        bad_len = copy.deepcopy(d)
        bad_len["log_wealth"].append(0.0)
        with self.assertRaises(ValueError):
            MeanEvidenceProcess.from_dict(bad_len)

        bad_t = copy.deepcopy(d)
        bad_t["t"] = -1
        with self.assertRaises(ValueError):
            MeanEvidenceProcess.from_dict(bad_t)

        bad_t_bool = copy.deepcopy(d)
        bad_t_bool["t"] = True
        with self.assertRaises(ValueError):
            MeanEvidenceProcess.from_dict(bad_t_bool)

        # t=0 状态：财富全为 1（log=0），却声称历史峰值 e^1000 —— 不一致
        bad_max = copy.deepcopy(fresh)
        bad_max["max_log_evidence"] = 1000.0
        with self.assertRaises(ValueError):
            MeanEvidenceProcess.from_dict(bad_max)

        # t=0 却声称 t=999 已停止
        bad_stop = copy.deepcopy(fresh)
        bad_stop["stop_time"] = 999
        with self.assertRaises(ValueError):
            MeanEvidenceProcess.from_dict(bad_stop)

        bad_nan = copy.deepcopy(d)
        bad_nan["m"] = float("nan")
        with self.assertRaises(ValueError):
            MeanEvidenceProcess.from_dict(bad_nan)

        bad_lam = copy.deepcopy(d)
        bad_lam["lambdas"][0] = 99.0  # > 1/m
        with self.assertRaises(ValueError):
            MeanEvidenceProcess.from_dict(bad_lam)

        with self.assertRaises(ValueError):
            MeanEvidenceProcess.from_json("{not json")

    def test_dead_track_serializes_as_null(self):
        p = MeanEvidenceProcess(0.5, [0.0, 2.0], 0.5)
        p.update(0)
        d = p.to_dict()
        self.assertEqual(d["log_wealth"], [0.0, None])
        r = MeanEvidenceProcess.from_dict(d)
        self.assertEqual(r.log_wealth(), (0.0, -math.inf))

    def test_serialize_then_restore_at_zero(self):
        p = MeanEvidenceProcess(0.25, [0.0, 4.0], 0.01)
        r = MeanEvidenceProcess.from_json(p.to_json())
        self.assertEqual(r.snapshot(), p.snapshot())
        self.assertEqual(r.t, 0)


class LongStreamNumericsTests(unittest.TestCase):
    def test_long_growth_stream_no_overflow(self):
        # 对数域：5000 步增长，E 远超 double 上界也不崩溃；
        # evidence 字段为 inf，log_evidence 仍有限。
        p = MeanEvidenceProcess(0.1, [0.0, 10.0], 0.05)
        p.extend([0.95] * 5000)
        s = p.snapshot()
        self.assertEqual(s.t, 5000)
        self.assertTrue(math.isfinite(s.log_evidence))
        self.assertEqual(s.evidence, math.inf)
        self.assertEqual(s.max_evidence, math.inf)
        self.assertEqual(s.p_value, 0.0)
        self.assertEqual(s.stop_time, 2)  # 前两步连续 0.95 即越过 20

    def test_mixed_long_stream_against_log_recurrence(self):
        # 独立于被测核心的另一参考：直接用 Python float 递推单轨道 log W。
        m, lam = 0.4, 2.0
        xs = [(i * 37 % 101) / 101.0 for i in range(300)]
        p = MeanEvidenceProcess(m, [lam], 0.1)
        log_w = 0.0
        dead = False
        for i, x in enumerate(xs, start=1):
            p.update(x)
            factor = 1.0 + lam * (x - m)
            if factor <= 0.0:
                dead = True
            if not dead:
                log_w += math.log(factor)
            snap = p.snapshot()
            if dead:
                self.assertEqual(snap.log_evidence, -math.inf)
            else:
                self.assertAlmostEqual(snap.log_evidence, log_w, places=12)


if __name__ == "__main__":
    unittest.main()
