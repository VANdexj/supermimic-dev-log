# G1 动作效果演示

成片：[showcase.mp4](showcase.mp4)。60 秒，1280×720，25 fps，H.264，原速播放，无音轨。

沿用原参考视频的呈现方式：实体为策略控制的 G1，蓝色半透明模型为参考动作，镜头跟随机器人。参考模型的水平位置锚定实体机器人，与原渲染脚本一致；因此该叠加主要用于观察姿态跟踪，不能据此判断全局位置误差。

使用最新一轮 `local_3060_v2` 训练中效果最好的已评估权重 `logs/local_3060_v2/ckpt_048000.pt`。最终保存的第 75,614 步权重回报为 0.7774、失败 4/40，所以选用第 48,000 步权重制作演示。

本次重新运行完整仿真录制，未使用参考动作替代策略轨迹。完整指标见 [recorded_eval.json](recorded_eval.json)：平均归一化回报 0.8157，失败 1/40（`jumps1_subject2`，未入选视频）。六段入选动作在此次完整评估中全部完成，视频只展示其中选定的十秒窗口。此前同一权重曾取得 0.8162、0/40 的复评结果，实际复评存在小幅波动。

| 视频时间 | 动作 | 原动作时间 | 完整动作回报 |
| --- | --- | --- | --- |
| 00–10 秒 | 舞蹈 `dance2_subject3` | 128–138 秒 | 0.8604 |
| 10–20 秒 | 格斗 `fight1_subject5` | 147–157 秒 | 0.8214 |
| 20–30 秒 | 跳跃 `jumps1_subject1` | 50–60 秒 | 0.8214 |
| 30–40 秒 | 冲刺 `sprint1_subject2` | 193–203 秒 | 0.8239 |
| 40–50 秒 | 倒地起身 `fallAndGetUp3_subject1` | 64–74 秒 | 0.7866 |
| 50–60 秒 | 舞蹈 `dance1_subject3` | 105–115 秒 | 0.8540 |

预览：[preview.jpg](preview.jpg)。生成脚本：[create_video.py](create_video.py)。剪辑时间与视频校验值：[manifest.json](manifest.json)。

在仓库根目录执行，使用已有 Docker 环境重新录制：

```bash
docker start supermimic
docker exec supermimic python supermimic/evaluate.py \
  --run_dir logs/local_3060_v2 \
  --ckpt logs/local_3060_v2/ckpt_048000.pt \
  --record results/local_3060_v2/showcase/trajectories.npz \
  --output results/local_3060_v2/showcase/recorded_eval.json
```

从已保存的策略轨迹重新生成预览和视频：

```bash
docker exec supermimic python results/local_3060_v2/showcase/create_video.py --preview
docker exec supermimic python results/local_3060_v2/showcase/create_video.py
```
