# 重训前失败分析

基线：`logs/local_3060/ckpt_023473.pt`，平均回报 0.5376，失败 20/40。

保存轨迹中，17 段触发高度误差阈值 0.25 m，3 段触发倾斜指标误差阈值 0.8。6 段起身动作全部失败。
倾斜指标为参考和机器人旋转矩阵 R[2,2] 之差的绝对值，不是角度（弧度）。

本次仿真转移约 376 万条；按仓库原始规模训练到 14,000 次迭代约为 1.44 亿条。
累计世界模型窗口和策略 rollout 数分别少约 95 倍、75 倍，支持训练规模不足的判断；尚不能排除其他训练问题。

回放缓冲区、FK 与状态重置检查通过；新运行保留网络、奖励和终止条件，校准批量后从头训练。

| 失败动作 | 首次失败（秒） | 完成比例 | 终止原因 |
|---|---:|---:|---|
| dance1_subject1 | 106.98 | 81.4% | tilt |
| dance2_subject1 | 64.90 | 28.8% | height |
| dance2_subject4 | 58.88 | 26.1% | height |
| dance2_subject5 | 139.30 | 61.7% | height |
| fallAndGetUp1_subject1 | 8.16 | 4.9% | height |
| fallAndGetUp1_subject4 | 6.86 | 4.1% | tilt |
| fallAndGetUp1_subject5 | 8.76 | 5.2% | height |
| fallAndGetUp2_subject2 | 15.12 | 9.2% | height |
| fallAndGetUp2_subject3 | 15.50 | 9.5% | height |
| fallAndGetUp3_subject1 | 10.54 | 10.3% | height |
| fight1_subject2 | 194.16 | 79.3% | height |
| fight1_subject3 | 75.40 | 30.8% | height |
| fightAndSports1_subject1 | 10.36 | 4.2% | height |
| jumps1_subject2 | 191.58 | 78.4% | height |
| run1_subject5 | 184.12 | 77.4% | height |
| walk2_subject1 | 194.94 | 81.9% | tilt |
| walk2_subject3 | 188.12 | 79.0% | height |
| walk3_subject1 | 186.82 | 75.8% | height |
| walk3_subject3 | 175.86 | 71.3% | height |
| walk3_subject4 | 218.60 | 88.6% | height |

完整数值见 [baseline_failure_analysis.json](baseline_failure_analysis.json)。
重训总预算 48 小时；仅当三个独立进程评估均达到平均回报 ≥0.82 且失败 0/40，才标记达标。
