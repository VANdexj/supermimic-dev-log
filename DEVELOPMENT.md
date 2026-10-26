# 个人开发记录

## 仓库来源

- 个人仓库：<https://github.com/VANdexj/supermimic-dev-log>（私有、独立仓库）。
- 原始仓库：<https://github.com/zita-ch/supermimic_vibed>。
- 初始基线提交：`a1508d42c9d1699a52e8233ab16d08c488441ab9`。
- 保留上游 Git 历史、作者署名和 [MIT 许可证](LICENSE)。

## 初始开发快照

将已有本地开发内容作为个人仓库的首个开发提交保存：

- RTX 3060 Laptop / 6 GB 显存的 Docker 环境及复现入口。
- 数据下载、训练、恢复、评估、结果导出与重训控制工具。
- 本地测试与 `results/local_3060*` 中可纳入 Git 的报告、指标、图片和 GIF。
- 当前结果见 [local_3060_v2/REPORT.md](results/local_3060_v2/REPORT.md)：
  选用第 48,000 次迭代权重，复评回报 0.8162，失败 0/40；未达到回报 ≥0.82 的目标。

沿用 `.gitignore`：下载数据、依赖目录、训练日志、模型权重、MP4 和轨迹 NPZ
保留在本地，不随 Git 提交。权重路径和校验信息见
[weights/README.md](results/local_3060_v2/weights/README.md)。

## 日常管理

`origin` 指向个人仓库，`upstream` 指向原始仓库，默认推送到 `origin`。

```bash
# 开始一项开发
git switch -c dev/your-topic

# 检查并保存本次改动
git status
git add <本次修改的文件>
git commit -m "说明本次开发内容"
git push -u origin HEAD

# 查看上游更新，按需决定是否合并
git fetch upstream
git log --oneline HEAD..upstream/main
```

后续实验可在本文追加记录，注明目的、配置、对应提交和结果目录。
代码变更使用 Git 提交追踪，实验指标以各次运行的报告为准。
