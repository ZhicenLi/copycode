# FedCorr + FedDS 融合工程

把 FedCorr 的**预处理阶段**（噪声客户端识别 + 标签修正）和 FedDS 的**可靠性加权聚合**接成一条流水线：前一段只吐修正后的标签，后一段用独立初始化的模型重新训练并按 Dawid-Skene 可靠性加权。

## 快速开始

```bash
pip install -r requirements.txt

# 1) 冒烟：合成数据，CPU，1 分钟
python run.py configs/smoke_cpu.json

# 2) 你的 idea 本身（只有 fedcorr+fedds 一条臂）
python run.py configs/idea_cifar10_k50.json

# 3) 改参数不用碰代码
python run.py configs/idea_cifar10_k50.json --fedds.rounds=5 --runtime.exp_name=debug
```

数据集首次运行时自动下载到 `data.data_root`；数据已就位就加 `--data.download=false`。

## 目录

```
fusion/
├── config.py            配置定义与校验（frac1、标注者数、数据集规格）
├── seeding.py           显式种子派生
├── analysis.py          与干净标签对照的诊断（识别率 / 纠错精确率）
├── data/
│   ├── specs.py         数据集规格（CIFAR-10/100 已实现，Tiny 预留）
│   ├── manifest.py      公共集→Dirichlet→注噪→校验→存取
│   └── loaders.py       索引式 Dataset、transform、缓存式评估
├── core/
│   ├── dawid_skene.py   DS 的 EM（log 空间，返回迭代数）
│   ├── gmm.py           两分量 GMM（sklearn 优先，numpy 兜底）
│   └── lid.py           LID（分块计算，K=10 也不会爆内存）
├── models/              ResNet18/34（CIFAR 版）、CNNCifar、模型工厂
├── stages/
│   ├── fedcorr_preprocess.py   链式预处理 + GMM 分组 + 修正
│   └── fedds_train.py          200 轮 + DS 加权（支持断点续跑）
├── pipeline.py          四条臂编排 + 断点复用
└── run.py               入口
```

## 四条实验臂

| arm | 训练标签 | 聚合 |
|---|---|---|
| `fedavg` | 原始带噪 | 按样本量加权 |
| `fedds` | 原始带噪 | DS 可靠性加权 |
| `fedcorr+fedavg` | 修正后 | 按样本量加权 |
| `fedcorr+fedds` | 修正后 | DS 可靠性加权 |

FedCorr 只跑一次，四条臂共用同一份修正标签（`fedcorr/corrected_labels.npy`），所以 2×2 表严格对齐。
只跑 idea 时把 `runtime.arms` 设成 `["fedcorr+fedds"]` 即可。

## 输出

```
outputs/<exp_name>/
├── config.json              完整配置快照
├── data/manifest.npz|.json  数据契约（含哈希校验）
├── data/noise_report.json   每客户端设定噪声率 vs 实际翻转率
├── fedcorr/
│   ├── corrected_labels.npy     修正后的标签（快照，不改 manifest）
│   ├── correction_meta.json     指纹，用于安全复用
│   ├── analysis.json            ★ 识别准确率、纠错精确率、每客户端估计 vs 真值
│   ├── fedcorr_iterations.jsonl 每轮：噪声客户端、估计噪声、累计 LID、准确率
│   └── fedcorr_clients.jsonl    每客户端：本地 loss、LID、μ
├── runs/<arm>/
│   ├── *_rounds.jsonl           每轮：DS 可靠性、迭代数、聚合权重、准确率
│   └── *_resume.pt              断点（仅 resume=true 时写）
└── summary.json|csv             结果表（含 fedcorr 诊断）
```

## 断点与复用

| 场景 | 做法 | 效果 |
|---|---|---|
| FedDS 跑到一半崩了 | 配置里 `runtime.resume=true` 后重跑 | 从最后一个 round 继续，结果与不中断完全一致 |
| 想换 FedDS 参数重跑 | 默认就复用 `corrected_labels.npy`（指纹匹配时） | 省掉几小时的 FedCorr |
| 改了 FedCorr 参数 | 指纹不匹配，自动重跑 FedCorr | 不会误用旧标签 |

## 与原文的差异（写论文时要交代）

1. **数据协议**：先切 10% 无标签公共集，再划分客户端（`data.partition="dirichlet"` 或 `"iid"`），再注噪（每客户端从 10%~100% 抽一档，翻转 `int(rate*n)` 个标签）。FedCorr 原文用的是 Bernoulli 系统噪声（每个客户端以概率 ρ 成为噪声客户端）。两种都能复现：`configs/ds_paper_cifar10.json` 用 IID + 逐位对齐的噪声率。
2. **只用 FedCorr 的 stage-1**：第二、三阶段不跑，因此 stage-2 之后的那次二次修正也不会发生。
3. **交叉熵默认在 logits 上算**：原文先把 softmax 结果喂给 `CrossEntropyLoss`（等于套了两层 softmax）。用 `fedcorr.loss_on="prob"` 可以切回原文行为。
4. **FedDS 阶段默认从头初始化**：原文 CIFAR-100 加载 ImageNet 预训练权重。用 `model.fedds_pretrained=true` 对齐原文。
5. **K=10 时每轮取 3 个客户端**（否则只有 1 个标注者，DS 在数学上失效）。K=50/100 保持 frac=0.1。可用 `fedds.frac_by_num_clients` 改。
6. **两阶段各用自家论文的数据管线**：`fedcorr.preprocess` 与 `fedds.preprocess` 独立。FedCorr 用 `standard`（RandomCrop+Flip+Normalize，配 ResNet18），FedDS 用 `raw`（x/255，配 CNNCifar）；评估时每个阶段用与自身训练一致的变换。
7. **公共集用确定性变换**：DS 要求所有客户端对同一批输入投票，随机裁剪会破坏这个前提。
8. **DS 在 log 空间计算**：与连乘数学等价，只是避免标注者变多后下溢；类先验默认关闭（与作者实现一致）。

## 已知成本

- 实测（RTX 4090，CIFAR-10，100 客户端，IID）：FedCorr 5 轮 × 5 epoch ≈ **27 分钟**；FedDS 100 轮 ≈ **25 分钟/臂**。
- 断点会存完整模型（ResNet18 约 45 MB/次），磁盘紧张时设 `runtime.checkpoint_models=false`。

## 测试

```bash
python -m unittest discover -s tests -v      # 75 个测试，约 30 秒
```

其中 7 个是**与官方源码对拍**（LID / ResNet18 / ResNet34 / CNNCifar / FedAvg / DS / 单标注者退化），
2 个是**FedDS 语义不变量**（客户端必须从同一全局模型出发），
1 个是**同种子可复现**，2 个是**断点续跑**（等价于不中断 / 改超参拒绝续跑），6 个是**诊断指标**。

## 文档

- `docs/DECISIONS.md`　　25 条编码决策（问题、选项、选择、改法）
- `docs/AUDIT.md`　　　　第一轮：与两篇原码逐项对照
- `docs/AUDIT_ROUND2.md`　第二轮：10 处问题与修复
- `docs/AUDIT_ROUND3.md`　第三轮：FedDS 客户端串联等 6 处问题
- `docs/AUTODL.md`　　　　AutoDL 部署要点
