"""固定起点权重混合证据过程的离线逐样本演示。

运行：python demo_iteration.py

纯标准库、无网络、无账号、无 sleep；所有数值在运行时实时计算。
"""

from __future__ import annotations

import time

from mean_evidence import ChangePointEvidenceProcess


M = 0.5
LAMBDAS = (0.0, 2.0)
ALPHA = 0.2
MAX_STARTS = 5
# 前三个可能起点给零权重；晚变化仍可被后面的固定权重检测。
# 权重在看数据前给定，总和为 1，之后不重分配。
WEIGHTS = (0.0, 0.0, 0.0, 0.5, 0.5)


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("-" * 78)


def print_snapshot(snap) -> None:
    contributions = ", ".join(f"{value:.4g}" for value in snap.contributions)
    components = ", ".join(f"{value:.4g}" for value in snap.component_evidence)
    print(
        f"t={snap.t:2d} 混合E={snap.evidence:8.4g} 峰值={snap.max_evidence:8.4g} "
        f"已开启={snap.active_starts} 首次报警={snap.stop_time}"
    )
    print(f"    各分量E: ({components})")
    print(f"    各起点贡献: ({contributions})")


def main() -> None:
    started = time.perf_counter()
    print("固定先验：m=0.5，lambda=(0,2)，alpha=0.2，阈值=1/alpha=5")
    print(f"最大起点数={MAX_STARTS}，固定权重={WEIGHTS}，权重和={sum(WEIGHTS):g}")

    banner("场景：前两点在 m 附近，随后持续高于 m 的晚变化")
    proc = ChangePointEvidenceProcess(
        M, LAMBDAS, ALPHA, WEIGHTS, MAX_STARTS
    )
    stream = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for x in stream:
        snap = proc.update(x)
        print_snapshot(snap)

    print(
        f"\n=> 混合证据首次越过阈值的时刻 t={snap.stop_time}；"
        "报警记录已冻结。"
    )
    print("继续输入一个低值：分量财富仍按原规则更新，权重不因报警而改变。")
    after_alarm = proc.update(0.0)
    print_snapshot(after_alarm)
    print(f"首次报警时刻仍为 {after_alarm.stop_time}，当前时间 t={after_alarm.t}。")

    banner("无变化对照：所有观测恰为 m=0.5，混合证据保持 1")
    flat = ChangePointEvidenceProcess(
        M, LAMBDAS, ALPHA, (0.1, 0.2, 0.3, 0.2, 0.2), MAX_STARTS
    )
    flat_snap = flat.extend([0.5] * MAX_STARTS)
    print_snapshot(flat_snap)

    print("\n解释范围：这是针对 H0: E[x_t | F_{t-1}] <= 0.5 的证据；")
    print("即使某个起点贡献最大，也不能把它解释成已认证的变点估计。")
    print(f"\n演示完成，实际耗时 {time.perf_counter() - started:.4f} 秒；全程离线。")


if __name__ == "__main__":
    main()
