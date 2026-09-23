"""固定输入的端到端演示（确定性、离线、纯标准库）。

运行：``python demo.py``

内容：
1. 正常结果：零假设成立的数据流下序贯监控，全程不拒绝；
2. 实际触发的拒绝：均值抬高的数据流，展示“可选停止”——首次越过
   1/alpha 即停，报停止时刻与 anytime p 值；
3. 边界：lambda=1/m 遇 x=0 的零因子（财富永久归零，不报错）；
4. 实际触发的异常：非法 lambda（>1/m）被拒绝，错误消息可定位；
5. 状态落盘恢复后继续追加，与未中断过程一致；恢复时改 m 被拒。

数据由固定种子的标准库 ``random.Random`` 生成，无网络、无附件。
"""

from __future__ import annotations

import random
import time

from mean_evidence import (
    InvalidParameterError,
    MeanEvidenceProcess,
    StateMismatchError,
)

M = 0.5
GRID = [i / 10.0 for i in range(0, 21)]  # 0,0.1,...,2.0；1/m=2 含端点
ALPHA = 0.05
SEED = 20260923
HORIZON = 200


def _stream(true_p: float) -> list[int]:
    rng = random.Random(SEED if true_p > M else SEED + 1)
    return [1 if rng.random() < true_p else 0 for _ in range(HORIZON)]


def _monitor(xs: list[int], true_p: float, label: str) -> tuple[int, MeanEvidenceProcess]:
    proc = MeanEvidenceProcess(M, GRID, ALPHA)
    print(
        f"--- {label}：真实 Bernoulli 参数 p={true_p}，零假设条件均值 <= m={M}，"
        f"alpha={ALPHA} ---"
    )
    for t, x in enumerate(xs, start=1):
        proc.append(x)
        if proc.stop_time is not None:
            print(
                f"t={t:3d}  x={x}  E_t={proc.evidence:.4g}  "
                f"anytime p={proc.p_value:.4g}  >= 阈值 1/alpha={1 / ALPHA:g}"
                "  → 首次拒绝，停止"
            )
            return t, proc
    print(
        f"t={proc.t:3d}（走完 {HORIZON} 步未越界）  E_t={proc.evidence:.4g}  "
        f"anytime p={proc.p_value:.4g}  → 不拒绝"
    )
    return -1, proc


def main() -> None:
    t0 = time.perf_counter()

    # 1) 正常结果：真实均值恰为 m，序贯监控不应（且本种子下没有）越界。
    null_data = _stream(0.5)
    print(f"[1] 零假设成立数据流：{sum(null_data)}/{HORIZON} 个 1")
    _monitor(null_data, 0.5, "正常监控（不拒绝）")
    print()

    # 2) 可选停止下实际触发拒绝：真实均值 0.72 > m。
    alt_data = _stream(0.72)
    print(f"[2] 均值抬高数据流：{sum(alt_data)}/{HORIZON} 个 1")
    stop_t, proc_alt = _monitor(alt_data, 0.72, "可选停止监控")
    assert stop_t > 0, "演示种子下应当触发拒绝"
    assert proc_alt.p_value <= ALPHA * (1 + 1e-12)
    # 停止之后继续喂数据：stop_time 锁定首次，p 值只可能更小。
    frozen = proc_alt.stop_time
    proc_alt.extend(alt_data[stop_t : stop_t + 5])
    print(f"    停止后再喂 5 个观测：stop_time 仍 = {proc_alt.stop_time}（== {frozen}），"
          f"anytime p={proc_alt.p_value:.4g}")
    print()

    # 3) 零因子边界：λ=1/m=2，x=0 时因子恰为 0。
    pz = MeanEvidenceProcess(M, [2.0], ALPHA)
    pz.append(0)
    print(f"[3] 零因子边界：λ=1/m=2 且 x=0 → log W={pz.log_wealth[0]}，E_t={pz.evidence}；")
    pz.append(1)
    print(f"    再喂 x=1 也不复活：log W={pz.log_wealth[0]}，E_t={pz.evidence}")
    print()

    # 4) 实际触发的异常：非法 lambda。
    print("[4] 非法输入边界：lambda=2.5 > 1/m=2")
    try:
        MeanEvidenceProcess(M, [0.0, 2.5], ALPHA)
    except InvalidParameterError as exc:
        print(f"    捕获 {type(exc).__name__}: {exc}")
    print()

    # 5) 序列化恢复 + 身份锁定（用对 m=0.5/0.51 都合法的网格，
    #    确保改 m 时撞上的是身份不一致而非网格越界）。
    grid5 = [0.0, 0.5, 0.8]
    p5 = MeanEvidenceProcess(M, grid5, ALPHA)
    p5.extend(alt_data[:30])
    blob = p5.to_json()
    restored = MeanEvidenceProcess.from_json(blob, m=M, lambda_grid=grid5, alpha=ALPHA)
    tail = alt_data[30:45]
    p5.extend(tail)
    restored.extend(tail)
    same = (
        p5.t == restored.t
        and p5.log_wealth == restored.log_wealth
        and p5.stop_time == restored.stop_time
    )
    print(f"[5] 在 t=30 序列化、恢复后继续喂 15 步：与未中断过程逐位一致 = {same}")
    try:
        MeanEvidenceProcess.from_json(blob, m=0.51, lambda_grid=grid5, alpha=ALPHA)
    except StateMismatchError as exc:
        print(f"    恢复时改 m 被拒：{type(exc).__name__}: {str(exc).split('；')[0]}")

    dt = time.perf_counter() - t0
    print()
    print(f"演示完成，实际计算耗时 {dt:.3f} 秒（无 sleep、无预录结果）。")
    print("结论口径：以上证据仅针对“条件均值 E[x_t|F_{t-1}] <= m”这一指定假设；")
    print("Ville 不等式保证可选停止下 P(越界) <= alpha，不要求独立性，也不推出其他结论。")


if __name__ == "__main__":
    main()
