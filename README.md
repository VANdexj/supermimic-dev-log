# SuperMimic — 个人开发与复现记录

本仓库用于保存个人开发修改、RTX 3060 Laptop 本机复现实验及结果记录。
计划托管于 `VANdexj/supermimic-dev-log`（待 GitHub 登录恢复后创建）。
基于 [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)，
独立维护并保留原始提交历史与 MIT 许可证。原项目介绍和作者署名保留如下。

开发记录与仓库管理方式见 [DEVELOPMENT.md](DEVELOPMENT.md)。

# SuperMimic reproduction on Isaac Lab 3 (kit-less, MuJoCo-Warp)

**Unofficial** reproduction of *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World
Model* (arXiv 2610.28480) — no official code is released, so everything here is implemented from
the paper. Target experiment: the controlled LAFAN1 comparison (Fig. 4a): 40 G1-retargeted LAFAN1
clips (2.45 h), one GPU; the paper reaches evaluation normalized return 0.82 after 1.2 h on an H100.

Author: **Chong Zhang** (ETH Zurich). License: [MIT](LICENSE).

> **Disclaimer:** this repository was speed-vibecoded in 1 hour with an AI coding
> assistant. It
> deliberately does not live under any lab's organization, because the code is too much "slop"
> for that: fast and unreviewed. Use it at
> your own risk.

> **One G1 motion tracker for all 40 LAFAN1 clips, trained in only ~1.5 GPU-hours**
> (1.47 h on a single RTX PRO 6000): evaluation return **0.829** with **0/40 failed clips**.
> For scale, the paper's PPO-in-simulator baseline does not reach return 0.82 within 18 h on up to
> 8 GPUs.

![SuperMimic G1 tracking LAFAN1 motions](results/supermimic_lafan1_highlight.gif)

*Solid robot: the trained policy simulated in Isaac Lab with MuJoCo-Warp. Blue ghost: the reference
motion. Clips: dance, jumps, fall-and-get-up, sprint.*

Progress and timestamps: [MILESTONES.md](MILESTONES.md).

## 本机复现：RTX 3060 Laptop / 6 GB 显存

此入口使用全部 40 段动作，保留网络、奖励、32 步想象及 8 步世界模型窗口，
缩小并行数和训练批量。正式训练默认最多运行 7200 秒，包含预热，达到时限后在当前迭代
结束时保存最终权重。首次构建、下载、短程验证和评估渲染另计。
本机短时训练的质量须以实际评估为准，README 上方的 0.829 / 0 次失败来自原作者硬件上的结果。

**环境隔离：** 宿主机只安装 Docker、NVIDIA Container Toolkit 及其必要组件。
Python、PyTorch、CUDA 工具链、编译器、ffmpeg 和其他项目依赖全部安装在镜像中；
不需要在宿主机执行 pip、conda 或创建 Python 虚拟环境。
项目目录仅存放源码、下载数据、缓存和结果，Docker 自身的镜像和构建缓存保存在 Docker 存储中。
首次构建建议预留至少 50 GB 磁盘空间（包含中间层和构建缓存），检查点另计：
每个 `ckpt_*.pt` 约 72 MB，`last.pt` 约 219 MB。

首次在本机终端执行以下命令（仅基础设施安装需要 sudo 密码）：

```bash
bash docker/install_runtime.sh
```

随后启动完整串行流程。`sg` 使新加入的 docker 组权限在当前会话生效；重新登录后可直接运行脚本。

```bash
sg docker -c './tools/reproduce.sh local_3060'
```

该脚本依次构建固定版本镜像、启动容器、下载并校验数据、转换机器人、运行 FK 与回放缓冲区检查，
执行包含策略和 critic 更新的短程训练，再验证断点恢复及时限保存、评估并渲染测试视频。
以上步骤通过后才开始正式训练，结束后串行评估至多三个阶段检查点（包含最终检查点），
按平均回报选择最佳检查点并导出实际策略视频。任何阶段失败都会停止。
已存在的同名运行不会被完整流程覆盖。

| 本机参数 | 初始配置 | 保守配置（`LOW_VRAM=1`） |
|---|---:|---:|
| 并行环境 | 64 | 16 |
| 缓冲步数 / 每环境 | 256 | 256 |
| 世界模型批量 / 每轮更新次数 | 64 / 2 | 16 / 2 |
| 策略批量 | 32 | 8 |
| 世界模型预热迭代 | 500 | 500 |
| 检查点间隔 | 200 | 200 |
| 精度 / torch.compile | bf16 / 关闭 | bf16 / 关闭 |

容器默认内存上限 10 GB、共享内存 2 GB；构建和 CPU 运算并发受限。
若初始配置显存不足，停止该次运行，用新名称和保守配置重新执行：

```bash
sg docker -c 'LOW_VRAM=1 ./tools/reproduce.sh local_3060_small'
```

环境和数据就绪后，可单独训练、恢复或重新导出结果：

```bash
./run_local.sh another_run
./run_local.sh local_3060 --resume logs/local_3060/last.pt --max_train_seconds 7200
docker exec supermimic python tools/finish_run.py logs/local_3060 --out results/local_3060
```

`--iterations` 表示累计目标迭代数，恢复时须大于检查点迭代数；恢复使用带优化器和 critic
状态的 `last.pt`，`ckpt_*.pt` 用于评估。每次恢复会重建并预填充回放缓冲区，
不保证与未中断训练逐步一致。每次调用的时间预算单独计时，训练日志记录累计训练时间。
所有实际参数写入 `logs/<run>/args.json`。

结果位于独立目录，不覆盖仓库示例：

- `logs/<run>/`：训练日志、检查点、阶段评估 JSON、`eval.csv`、镜像 ID、GPU 信息与依赖版本。
- `results/<run>/REPORT.md`：本次指标、目标是否达到和视频预览。
- `results/<run>/summary.json`：最佳检查点、实际配置、全部 40 段动作的回报及失败列表。
- `results/<run>/success_curve.png`、`highlight.mp4`、`highlight.gif`、`trajectories.npz`。

仅当本次实测平均回报 ≥0.82 且失败 0/40 时，报告才标记达到完整指标。
失败动作会在视频中标注 terminated；视频来自策略仿真轨迹。

### 本次实测（RTX 3060 Laptop，6 GB）

运行 `local_3060`：使用上表初始配置，正式训练 7200 秒，最终保存于第 23,473 次迭代。
所有评估均使用全部 40 段动作、固定摩擦系数 1.0、关闭随机化和动作/观测噪声。

| 检查点迭代 | 累计训练时间 | 平均归一化回报 | 失败动作 |
|---:|---:|---:|---:|
| 8,000 | 0.663 h | 0.4565 | 18/40 |
| 15,800 | 1.337 h | 0.5080 | 19/40 |
| 23,473 | 2.000 h | 0.5376 | 20/40 |

**本次完成了容器内训练、恢复、评估及渲染闭环，但未复现 README 的完整指标和跟踪质量。**
该结果使用小批量配置，不能等同于上方原作者的 4096 环境实验。
训练时观测到显存约 2.3 GB、容器内存约 2.7 GB；所有项目依赖均在 Docker 内。

实际产物：[结果报告](results/local_3060/REPORT.md)、[逐段指标与配置](results/local_3060/summary.json)、
[40 秒 MP4](results/local_3060/highlight.mp4)、[GIF](results/local_3060/highlight.gif)、
[曲线](results/local_3060/success_curve.png)、[四类动作预览](results/local_3060/preview.png)。
完整运行记录在 `logs/local_3060/reproduction.log`；恢复点为 `logs/local_3060/last.pt`。

## 本机长程重训（48 小时总预算）

已有容器和数据就绪后，使用新的运行名启动自动校准和重训：

```bash
sg docker -c './run_retrain.sh local_3060_v2'
tail -f logs/local_3060_v2.controller.log
cat logs/local_3060_v2/controller_state.json
```

控制器在容器内的同名 tmux 会话中运行，终端关闭后继续执行。总预算从控制器启动计时，
包括检查、校准、训练、评估和导出，默认最多 48 小时；达到截止时间不再启动新的训练段。
默认每段训练最多 8 小时，并为最终评估和导出预留 1 小时。
这是一轮受时间限制的达标实验，不能预先保证 6 GB 显存设备达到仓库指标。

流程先检查 FK、回放缓冲区和状态重置，再分别校准两组配置：

| 参数 | 配置一 | 配置二 |
|---|---:|---:|
| 并行环境 | 256 | 128 |
| 缓冲步数 / 每环境 | 1000 | 1000 |
| 世界模型 batch / 每轮更新次数 | 128 / 10 | 64 / 10 |
| 策略 batch | 256 | 128 |

每组校准默认最多 200 次迭代或 900 秒，包含策略和 critic 更新；要求指标有限、
采样 GPU 占用和 PyTorch 峰值预留显存均不超过 5 GiB、容器常驻匿名/共享内存不超过 7 GiB。
从通过检查的配置中选择仿真转移吞吐最高者。两组都不通过则停止。
通过恢复、时限保存和真实策略视频导出检查后，正式模型从头训练，不使用校准权重。

网络、奖励、终止条件、随机化、bf16、32 步想象及 8 步世界模型窗口保持不变，关闭编译。
按 `s = 4096 / num_envs` 设置世界模型预热 `500*s` 次迭代和学习率衰减 `250000*s` 次迭代；
初始化及每次恢复均采集 1000 步填充回放。该调整对齐样本量尺度，小批量优化过程仍不等同于原实验。

每 2000 次迭代保存恢复状态，每段结束串行评估新增的每 8000 次迭代候选及段末权重，
全部评估 40 段完整动作。报告增加完成比例、首次失败时间、终止原因及高度/倾斜误差。
同时保留最高回报、最低失败数及满足联合指标的模型；只有三个独立进程评估均达到
平均回报 ≥0.82 且失败 0/40，才标记 `target_reached: true`。

阶段汇总写入 `results/<run>/summary.json`，最终导出曲线、完整轨迹、MP4 和 GIF。
控制器仅清理新运行中不再需要的普通快照和未被保留的已评估权重；保留阶段评估 JSON、
最近两份权重、关键模型及待评估候选。原有运行不做清理。
环境版本、数据来源、代码快照和本地 diff 保存在新日志目录。

控制器异常退出后，可在原截止时间内恢复；恢复不会重新获得 48 小时预算：

```bash
sg docker -c './run_retrain.sh local_3060_v2 --resume'
```

`tools/finish_run.py` 现在默认评估每 8000 次迭代候选和最终权重，复用已保存评估，
保留完整历史，并使用选中检查点自身的配置。`--metrics_only` 只生成阶段指标；
`supermimic/evaluate.py --output <path>` 可把独立验证写入另一文件，保留原评分记录。

## Layout

| Path | What |
|---|---|
| `IsaacLab/` | Isaac Lab `main` @ 46d6cc3 (VERSION 3.0.0), unmodified — cloned by `tools/setup.sh`, not tracked |
| `docker/` | `Dockerfile`: `supermimic:latest` = `isaac-lab-kitless:main-46d6cc3` (built from `IsaacLab/docker/Dockerfile.kitless`) + tmux, ffmpeg, gcc (for `torch.compile`); `run_container.sh` |
| `data/` | not tracked; `tools/download_data.sh` fetches LAFAN1 G1 retargets (HF `lvhaidong/LAFAN1_Retargeting_Dataset`) and the G1 URDF/meshes, then builds `g1_29dof.xml` (MJCF) and the USD |
| `supermimic/` | implementation (see below) |
| `tools/` | setup, asset conversion, tests, benchmark, plotting, rendering, failure analysis |
| `logs/<run>/` | not tracked: `train_log.csv`, `eval.csv`, checkpoints, stdout |
| `results/` | success curve and highlight GIF (full videos are rendered locally with `tools/render.py`) |

## Setup

Neither Isaac Lab nor any data is included in this repository; the steps below fetch them.

**Requirements:** Linux, an NVIDIA GPU (tested: RTX PRO 6000 Blackwell, driver 580 / CUDA 13),
Docker with the NVIDIA Container Toolkit (`docker run --gpus` must work), ~20 GB of disk for the
images and ~0.2 GB for the data.

Physics: Newton with the MuJoCo-Warp solver (`NewtonCfg(MJWarpSolverCfg(...))`), no Kit, no PhysX;
50 Hz control, 4 substeps of 5 ms, implicitfast integrator, MuJoCo collision pipeline.

### 1. Isaac Lab (kit-less Docker image)

The code runs against Isaac Lab 3 `main` at commit
[`46d6cc3`](https://github.com/isaac-sim/IsaacLab/commit/46d6cc3d7b115906fad12efd372fbf69f4e3d3ed)
(VERSION 3.0.0), unmodified, inside Isaac Lab's **kit-less** image (no Isaac Sim / Kit, no PhysX).
Newer `main` commits may work but are untested.

```bash
tools/setup.sh
```

which is equivalent to:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
git -C IsaacLab checkout 46d6cc3d7b115906fad12efd372fbf69f4e3d3ed
(cd IsaacLab && docker build -f docker/Dockerfile.kitless -t isaac-lab-kitless:main-46d6cc3 .)
docker build -t supermimic:latest docker/     # + tmux, ffmpeg, gcc, NVIDIA EGL vendor file
```

Then start the long-lived container (repository mounted at `/workspace/supermimic`):

```bash
docker/run_container.sh 0 128g     # GPU index 0, 128 GB memory cap
```

### 2. Data: LAFAN1 motions and the G1 model

| What | Source | License | Goes to |
|---|---|---|---|
| 40 LAFAN1 clips retargeted to the 29-DoF G1 (30 fps CSV: root pos, root quat xyzw, 29 joint angles) | HF [`lvhaidong/LAFAN1_Retargeting_Dataset`](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset), folder `g1/` | CC BY-NC-ND 4.0 (Ubisoft La Forge) | `data/lafan1_g1/*.csv` |
| G1 description `g1_29dof_rev_1_0.urdf` + meshes (the model the clips were retargeted to) | same dataset, folder `robot_description/g1/` | BSD-3-Clause (Unitree) | `data/g1_description/` |

With the container from step 1 running:

```bash
tools/download_data.sh
```

It downloads both, then converts the robot inside the container:
`tools/urdf_to_mjcf.py` compiles the URDF into `data/g1_description/g1_29dof.xml` (free-floating
pelvis), and `tools/mjcf_to_usd.py` converts that MJCF to `data/g1_usd/g1_29dof/g1_29dof.usda`,
which Isaac Lab spawns. The conversion is offline because the kit-less importers cannot run inside
an Isaac Lab process (see Deviations). Expected result:

```
data/
├── lafan1_g1/                  40 CSVs (dance, fallAndGetUp, fight, jumps, run, sprint, walk)
├── g1_description/
│   ├── g1_29dof_rev_1_0.urdf
│   ├── g1_29dof.xml            generated MJCF
│   └── meshes/*.STL            38 meshes
└── g1_usd/g1_29dof/            generated USD (+ payloads/)
```

Check: `ls data/lafan1_g1 | wc -l` prints `40`, and the clips total 264,705 frames at 30 fps
(2.45 h). `tools/test_env.py` checks the converted robot against the reference motions (FK vs.
simulator should agree to ~1e-6):

```bash
docker exec supermimic python tools/test_env.py
```

## Usage

```bash
CPUS=0-23 ./run_train.sh lafan1_main          # training in tmux session 'lafan1_main' (optional CPU pinning)
docker exec -it supermimic tmux attach -t lafan1_main
docker exec supermimic tmux new -d -s eval_main "cd /workspace/supermimic && python supermimic/evaluate.py --run_dir logs/lafan1_main --watch"
docker exec supermimic python tools/plot.py logs/lafan1_main --out results/success_curve.png
docker exec supermimic python supermimic/evaluate.py --run_dir logs/lafan1_main --ckpt logs/lafan1_main/ckpt_014000.pt --record traj.npz
docker exec supermimic python tools/render.py traj.npz out.mp4 --clips dance2_subject1 --seconds 10
```

The first iterations are slow while `torch.compile` builds its kernels (cached in `.cache/`).
Checkpoints are written every 2000 iterations to `logs/<run>/`; the eval watcher scores each on all
40 clips and appends to `logs/<run>/eval.csv`.

## Result (LAFAN1, 40 clips, one RTX PRO 6000 Blackwell)

Evaluation return 0.824 at 1.24 h of training (paper: 0.82 at 1.2 h on an H100) and 0.829 with
**0/40 failed clips** at 1.47 h (iteration 14000).

![success curve](results/success_curve.png)

## Implementation (paper section -> file)

* G1 29-DoF, BeyondMimic actuators (10 Hz natural frequency, damping ratio 2, motor armatures,
  action scale 0.25·τmax/Kp) — `supermimic/robot.py`.
* Reference library: 30 -> 50 Hz resampling, finite-difference velocities, FK of 14 tracked bodies,
  1 s bins — `supermimic/motion.py`, differentiable torch FK `supermimic/kinematics.py`
  (matches the simulator's link poses/velocities to 1e-6).
* MDP (App. A): observation (q−q0, q̇, IMU ω, projected gravity, a_{t−1}; 10-frame history; Table A.1
  noise), K=10 future reference frames as the goal, BeyondMimic tracking reward without the
  self-collision term and joint-limit weight −2, termination |e_z|>0.25 m or e_tilt>0.8, sim context
  ξ = (friction U[0.3,1.2], joint offsets U[±0.01], torso CoM offset) — `supermimic/mdp.py`.
* Simulator wrapper with return-based reset sampling (50% uniform / 40% softmax(−25 G_b/G_max) /
  10% past-softmax) — `supermimic/env.py`.
* World model f_ω: MLP(2048,2048,2048) on heading-invariant state features + action + ξ, predicting
  frame-local increments; fit on k=8-step autoregressive windows (Eq. 1); analytic reward,
  observation and termination heads — `supermimic/networks.py`, `supermimic/train.py`.
* Policy: motion encoder MLP(2048,1024,512,256)->z∈R^32 (tanh), shared decoder
  MLP(2048,1024,512,256); critic MLP(256,128). BPTT over H=32 imagined steps from replay-buffer
  states with a soft survival weight and critic bootstrap (Eq. 2); critic on GAE(λ=0.95) targets
  (Eq. 3), 5 epochs × 4 minibatches.
* Hyperparameters as Table A.5: 4096 envs, buffer 1000 steps/env, action noise N(0, 0.1²),
  UTD 0.4 (2.5 vectorized sim steps per iteration), 10 WM batches of 2048 windows per iteration,
  4096 imagined rollouts, Adam lr 2e-4 (policy β=(0.9,0.95)), cosine decay to 0.03× over 250k
  iterations, grad clip 1.0 (+ element-wise 0.1 for the policy), γ=0.99.
* Evaluation (App. F.1): each clip once from its first frame, friction 1.0, no DR/noise,
  deterministic policy; failure = training termination — `supermimic/evaluate.py`.

## Deviations / choices not specified by the paper

* **Self-collisions disabled** in the simulator (speed); robot–ground contacts use the URDF's
  convex mesh hulls and foot spheres.
* **bf16 autocast for MLP matmuls** (paper: TF32). On this RTX PRO 6000 it is 24 % faster; all
  state integration, FK and rewards stay fp32.
* Reward, observation and termination heads are **analytic** (allowed by §3.2); termination in
  imagination is a sigmoid-softened version of the hard thresholds.
* The policy observation additionally contains the reference root orientation relative to the
  robot root (6D), i.e. BeyondMimic's `motion_anchor_ori_b` — IMU-observable; no global position.
* 500 world-model-only warm-up iterations before policy updates start.
* Torso-CoM and friction are randomized once per environment at start-up (BeyondMimic's
  start-up events); joint offsets are resampled every episode. Episodes time out after 10 s or at
  the clip end (bootstrapped, not terminal).
* Do not run under a virtual-memory `ulimit -v`: Isaac Lab + Warp reserve ~430 GB of *virtual*
  address space (CUDA VA reservations; RSS ~3 GB) and Warp's CUDA-graph capture needs its mempool,
  which cannot initialize under such a limit. Cap memory with the container's cgroup
  (`--memory`) instead. Rootless docker ignores `--cpuset-cpus`; pin CPUs with `taskset`
  (`CPUS=...` in `run_train.sh`).
* Asset import: the kit-less URDF/MJCF importers fail inside a running Isaac Lab process
  (`PhysxJointAPI` schema not registered), so the USD is converted once, offline, by
  `tools/mjcf_to_usd.py` (registers the importer's schema plugin first) and spawned with
  `UsdFileCfg(variants={"Physics": "mujoco"})`.

## Attribution and licenses

* Method: Q. Liao, Y. Shao, J. Long, K. Zakka, R. Horowitz, P. Abbeel, K. Sreenath,
  *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*,
  CoRL 2026, arXiv:2610.28480. This code is an independent re-implementation, not affiliated with
  the authors.
* Isaac Lab (BSD-3-Clause), Newton and MuJoCo-Warp (Apache-2.0) are used unmodified.
* LAFAN1 motions (Ubisoft La Forge): CC BY-NC-ND 4.0; G1 retargets and description by Unitree
  (BSD-3-Clause). The motion data is not redistributed here; the demo GIF shows the
  reference motions for illustration only and is not covered by this repository's MIT license.

## License

The code in this repository is released under the [MIT License](LICENSE).

Copyright (c) 2026 ETH Zurich, Chong Zhang.
