"""状态序列化 / 恢复 / 参数身份锁定测试。"""

from __future__ import annotations

import json
import math
import unittest

from mean_evidence import (
    InvalidParameterError,
    MeanEvidenceProcess,
    SerializationError,
    StateMismatchError,
)

M = 0.5
GRID = [0.0, 0.8, 2.0]
ALPHA = 0.1


class SerializationTests(unittest.TestCase):
    def _drive(self, p: MeanEvidenceProcess) -> None:
        p.extend([0.9, 0.95, 0.1, 0.8, 0.9, 0.0])

    def test_dict_roundtrip_preserves_everything(self) -> None:
        p = MeanEvidenceProcess(M, GRID, ALPHA)
        self._drive(p)
        state = p.to_dict()

        q = MeanEvidenceProcess.from_dict(state, m=M, lambda_grid=GRID, alpha=ALPHA)
        self.assertEqual(q.t, p.t)
        self.assertEqual(q.m, p.m)
        self.assertEqual(q.lambda_grid, p.lambda_grid)
        self.assertEqual(q.alpha, p.alpha)
        for a, b in zip(q.log_wealth, p.log_wealth):
            self.assertEqual(a, b)
        self.assertEqual(q.log_evidence, p.log_evidence)
        self.assertEqual(q.max_log_evidence, p.max_log_evidence)
        self.assertEqual(q.stop_time, p.stop_time)

        # 恢复后继续追加，与未中断的原过程逐位一致。
        more = [0.9, 0.3, 1.0, 0.0]
        p.extend(more)
        q.extend(more)
        self.assertEqual(q.t, p.t)
        for a, b in zip(q.log_wealth, p.log_wealth):
            self.assertEqual(a, b)
        self.assertEqual(q.stop_time, p.stop_time)

    def test_json_roundtrip(self) -> None:
        p = MeanEvidenceProcess(M, GRID, ALPHA)
        p.extend([0.9, 0.0, 0.9])
        text = p.to_json()
        # 不允许 NaN/Infinity 字面量进入 JSON。
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        q = MeanEvidenceProcess.from_json(text, m=M, lambda_grid=GRID, alpha=ALPHA)
        self.assertEqual(q.log_wealth, p.log_wealth)
        self.assertEqual(q.t, 3)

    def test_roundtrip_with_zero_factor(self) -> None:
        p = MeanEvidenceProcess(M, GRID, ALPHA)
        p.append(0.0)  # lambda=2 死亡
        self.assertEqual(p.log_wealth[2], -math.inf)
        q = MeanEvidenceProcess.from_json(
            p.to_json(), m=M, lambda_grid=GRID, alpha=ALPHA
        )
        self.assertEqual(q.log_wealth, p.log_wealth)
        q.append(1.0)
        p.append(1.0)
        self.assertEqual(q.log_wealth, p.log_wealth)
        self.assertEqual(q.log_wealth[2], -math.inf)

    def test_restore_rejects_changed_m(self) -> None:
        # 网格对两个 m 都合法（0.8 <= 1/0.51），确保撞上的是身份不一致，
        # 而不是新 m 下的网格越界参数校验。
        grid = [0.0, 0.8]
        p = MeanEvidenceProcess(M, grid, ALPHA)
        p.extend([0.9, 0.2])
        state = p.to_dict()
        with self.assertRaises(StateMismatchError):
            MeanEvidenceProcess.from_dict(state, m=0.51, lambda_grid=grid, alpha=ALPHA)
        with self.assertRaises(InvalidParameterError):
            MeanEvidenceProcess.from_dict(state, m=0.5)  # 缺 grid/alpha

    def test_restore_rejects_changed_grid(self) -> None:
        p = MeanEvidenceProcess(M, GRID, ALPHA)
        state = p.to_dict()
        with self.assertRaises(StateMismatchError):
            MeanEvidenceProcess.from_dict(
                state, m=M, lambda_grid=[0.0, 0.8, 2.0, 0.1], alpha=ALPHA
            )
        with self.assertRaises(StateMismatchError):
            MeanEvidenceProcess.from_dict(
                state, m=M, lambda_grid=[0.0, 0.81, 2.0], alpha=ALPHA
            )
        # 顺序变化也算不同网格。
        with self.assertRaises(StateMismatchError):
            MeanEvidenceProcess.from_dict(
                state, m=M, lambda_grid=[2.0, 0.8, 0.0], alpha=ALPHA
            )

    def test_restore_rejects_changed_alpha(self) -> None:
        p = MeanEvidenceProcess(M, GRID, ALPHA)
        state = p.to_dict()
        with self.assertRaises(StateMismatchError):
            MeanEvidenceProcess.from_dict(state, m=M, lambda_grid=GRID, alpha=0.05)

    def test_corrupt_states_rejected(self) -> None:
        p = MeanEvidenceProcess(M, GRID, ALPHA)
        p.extend([0.9, 0.1])
        good = p.to_dict()

        bad_version = dict(good, version=99)
        with self.assertRaises(SerializationError):
            MeanEvidenceProcess.from_dict(bad_version, m=M, lambda_grid=GRID, alpha=ALPHA)

        bad_len = dict(good, log_wealth=good["log_wealth"][:2])
        with self.assertRaises(SerializationError):
            MeanEvidenceProcess.from_dict(bad_len, m=M, lambda_grid=GRID, alpha=ALPHA)

        bad_t = dict(good, t=-1)
        with self.assertRaises(SerializationError):
            MeanEvidenceProcess.from_dict(bad_t, m=M, lambda_grid=GRID, alpha=ALPHA)

        bad_stop = dict(good, stop_time=99)
        with self.assertRaises(SerializationError):
            MeanEvidenceProcess.from_dict(bad_stop, m=M, lambda_grid=GRID, alpha=ALPHA)

        bad_dead = dict(good, dead=[True, False, False])
        # 死亡项的 log_wealth 必须为 null。
        with self.assertRaises(SerializationError):
            MeanEvidenceProcess.from_dict(bad_dead, m=M, lambda_grid=GRID, alpha=ALPHA)

        bad_json = "{not json"
        with self.assertRaises(SerializationError):
            MeanEvidenceProcess.from_json(bad_json, m=M, lambda_grid=GRID, alpha=ALPHA)

    def test_internally_inconsistent_states_rejected(self) -> None:
        p = MeanEvidenceProcess(M, GRID, ALPHA)
        p.extend([0.9, 0.9, 0.1])
        good = p.to_dict()

        # 运行最大值低于当前证据：矛盾。
        bad_max = dict(good, max_log_evidence=-5.0)
        with self.assertRaises(SerializationError):
            MeanEvidenceProcess.from_dict(bad_max, m=M, lambda_grid=GRID, alpha=ALPHA)

        # 已越过阈值却声称从未停止：矛盾（阈值 -log 0.1 ≈ 2.3026）。
        bad_stop = dict(good, max_log_evidence=5.0, stop_time=None)
        with self.assertRaises(SerializationError):
            MeanEvidenceProcess.from_dict(bad_stop, m=M, lambda_grid=GRID, alpha=ALPHA)

        # 声称停止过但最大值不到阈值：矛盾。
        bad_stop2 = dict(good, stop_time=2)
        with self.assertRaises(SerializationError):
            MeanEvidenceProcess.from_dict(bad_stop2, m=M, lambda_grid=GRID, alpha=ALPHA)

    def test_json_constants_rejected(self) -> None:
        p = MeanEvidenceProcess(M, GRID, ALPHA)
        text = p.to_json()
        # 手工塞入 NaN 常量。
        tampered = text.replace(
            '"max_log_evidence": 0.0', '"max_log_evidence": NaN', 1
        )
        if tampered == text:  # 兜底：直接构造
            data = json.loads(text)
            data["max_log_evidence"] = float("nan")
            tampered = json.dumps(data)  # 默认 allow_nan 产生 NaN 字面量
        with self.assertRaises(SerializationError):
            MeanEvidenceProcess.from_json(tampered, m=M, lambda_grid=GRID, alpha=ALPHA)

    def test_append_after_restore_equals_uninterrupted(self) -> None:
        # 停止发生后再序列化：stop_time 必须保留。
        p = MeanEvidenceProcess(0.3, [0.0, 1.0, 1.0 / 0.3], 0.2)
        for x in [0.95] * 40:
            p.append(x)
        self.assertIsNotNone(p.stop_time)
        q = MeanEvidenceProcess.from_dict(
            p.to_dict(), m=0.3, lambda_grid=[0.0, 1.0, 1.0 / 0.3], alpha=0.2
        )
        self.assertEqual(q.stop_time, p.stop_time)
        self.assertTrue(math.isclose(q.p_value, p.p_value, rel_tol=1e-15))


if __name__ == "__main__":
    unittest.main()
