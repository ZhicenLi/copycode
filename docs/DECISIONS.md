# 编码决策记录（Decision Log）

> 版本：v1.0　日期：2026-10-18
> 范围：`fusion/` 工程从零到"可运行"过程中，所有影响行为的决定。
> 用法：每条决策都有编号、你的选择、理由、代码位置、当前状态、以及"想改怎么改"。写论文的"实现细节"一节可以直接引用编号。

---

## 0. 总原则（你反复强调的）

> **"我的目标不是保留算法，我的目标是最终效果好，证明我的 idea 好，才能发文章。"**

由此定下三条准则，后面所有决策都是在它们之间权衡：

| 准则 | 含义 | 冲突时的取舍 |
|---|---|---|
| **A. 结果可信** | 算法主干必须和原文一致，否则结果不可解释 | 优先 |
| **B. 结果能打** | 明显压制性能的实现缺陷要修 | 次优先 |
| **C. 可回退** | 任何偏离都留开关，能切回原文 | 必须满足 |

---

## 1. 架构类

### D01　不依赖 PFLlib 源码，只沿用其骨架风格

- **问题**：PFLlib 的数据层是"每客户端一个 npz"，而我们需要"先切公共集→Dirichlet→注噪→标签交接"的动态协议。
- **选项**：A 真依赖 PFLlib，改造其数据层；B 沿用其 Server/Client 分层与命名，自建轻量实现。
- **你的选择**：**B**。
- **理由**：改造 PFLlib 数据层会把它改得面目全非，且引入大量与本研究无关的代码。
- **位置**：整个 `fusion/` 包；`stages/` 对应 Server，`FedDSTrainer._local_train` 对应 Client。
- **状态**：已实现。
- **影响**：⚠️ 若导师硬性要求"必须基于 PFLlib"，需要迁移。

### D02　manifest 是唯一真源，样本编号不重排

- **问题**：两阶段必须看到同一批样本，且标签交接要可追溯。
- **选择**：一张表记录 `sample_id / client_id / clean_label / noisy_label / corrected_label / noise_rate / split`，`sample_id` 是原始训练集下标，永不重编号；文件带 SHA-256。
- **理由**：编号一旦重排，"同一条样本"就失去意义，交接和审计都会失效。
- **位置**：`fusion/data/manifest.py`。
- **状态**：已实现，测试 `test_conservation` 覆盖。

### D03　FedCorr 预处理保持"链式"语义

- **问题**：官方代码 `int(K*frac1)=1` 时，`FedAvg` 只聚合一个客户端，等价于"把全局模型替换成该客户端的模型"。
- **选项**：A 修正成真正的联邦平均；B 保持原样。
- **你的选择**：**B（按我的建议）**。
- **理由**：改成联邦平均就不是 FedCorr 了，结果无法与原文对比。
- **位置**：`stages/fedcorr_preprocess.py::run()`，注释里写明"one client per aggregation step"。
- **状态**：已实现。
- **影响**：⚠️ 这一点必须在论文里说明，否则读者会以为是标准 FL。

### D04　两阶段只传标签，模型完全隔离

- **选择**：FedCorr 只输出 `corrected_labels.npy`；FedDS 独立初始化自己的模型、优化器。
- **位置**：`pipeline.py`；FedDS 阶段从不接收 FedCorr 的 `state_dict`。
- **状态**：已实现。

### D05　配置用 JSON，支持命令行覆盖

- **问题**：本地没有 PyYAML。
- **选择**：JSON + `--section.key=value` 覆盖；同时保留 YAML 读取分支（装了 PyYAML 才可用）。
- **位置**：`fusion/config.py::_read_structured / parse_cli`。
- **状态**：已实现。

---

## 2. 实验参数类

### D06　K=10 时把 frac 提到 0.3（每轮 3 个标注者）

- **问题**：FedDS 的 `m = max(int(frac*K),1)`。frac=0.1 时 K=10 → m=1，DS 只有一个标注者，混淆矩阵退化成单位阵，可靠性恒为 1，**等价于 FedAvg**。
- **选项**：A 照跑接受退化；B K=10 用 0.3；C K=10 不做 FedDS；D 解耦"参与"与"标注"。
- **你的选择**：**B**。
- **理由**：每组至少要有 3 个标注者，DS 才有信息；同时保留可解释的偏离记录。
- **位置**：`config.py::FedDSConfig.frac_by_num_clients = {10: 0.3}`、`clients_per_round()`。
- **实测**：K=10/50/100 → 3/5/10 个标注者（测试 `test_annotator_floor_per_client_count`）。
- **状态**：已实现。
- **改法**：改 `frac_by_num_clients` 或 `fedds.frac`。

### D07　CIFAR-100 的 FedDS 模型默认**不用** ImageNet 预训练

- **问题**：官方 `ResNet18Cifar100` 写死 `weights=IMAGENET1K_V1`，与"独立初始化"和与 FedCorr 阶段的可比性冲突。
- **选项**：A 允许预训练；B 全部从头；C 默认 B + 开关。
- **你的选择**：**C（"按你的来，我要效果好"）**。
- **位置**：`config.py::ModelConfig.fedds_pretrained = False`；`models/registry.py::_resnet18_torchvision`。
- **状态**：已实现。`pretrained=False` 时走本工程自己的 ResNet（本地与服务器行为完全一致）；`True` 时才需要 torchvision 和联网。
- **影响**：主表 CIFAR-100 的绝对值会低于论文，需在论文里注明。

### D08　Tiny-ImageNet 本轮不做，只留接口

- **你的选择**：三个数据集迟早都要做，但可以后期再加。
- **位置**：`data/specs.py` 已登记 `tinyimagenet`（200 类、64×64、`implemented=False`）；配置层会直接拒绝。
- **状态**：**预留**。加的时候需要：ImageFolder 加载器、64×64 的模型适配、LID 内存评估。

### D09　FedDS 阶段强制/默认 200 轮

- **位置**：`FedDSConfig.rounds = 200`；所有生成配置都是 200。
- **状态**：已实现（可覆盖，但论文主表用默认）。

### D10　local_ep 两档：FedDS 阶段 1 / 5

- **位置**：`fedds.local_ep`；生成的 24 份配置里各一半。
- **注意**：**不反向影响 FedCorr 阶段**——FedCorr 的 `local_ep` 固定按原文（5）。

---

## 3. 算法细节类（当初的"六个技术疑惑"）

### D11　FedCorr 打分时保留随机变换

- **问题**：原代码用带 `RandomCrop`/`Flip` 的训练变换去算 LID 和 loss，同一张图每轮的打分都不同。
- **选项**：保留（忠实）／改成确定性（更干净）。
- **你的选择**：**按建议保留**。
- **理由**：GMM 的分组阈值是在这个"带噪打分"上调出来的，改了分布会漂。
- **位置**：`config.py::FedCorrConfig.train_transform = "train"`。
- **状态**：已实现，可切 `"eval"` 对比。

### D12　交叉熵算在 logits 上（修正官方双层 softmax）

- **问题**：官方 `get_output()` 先 `softmax`，再把概率喂给 `CrossEntropyLoss`，等于套了两层 softmax，loss 被压扁、区分度下降。
- **选项**：A 保留（忠实）；B 修正为正常 CE（更好）；C 默认 B + 开关。
- **你的回答**：**"我看不懂，你用最好的解决办法。"**
- **最终决定**：**默认 B（修正），保留开关切回 A**。
- **理由**：结合总原则 B（结果能打），这是明显缺陷，不修只会让标签修正变差；而修正只影响损失的尺度与排序，不改变算法结构。
- **位置**：`config.py::FedCorrConfig.loss_on = "logit"`；`stages/fedcorr_preprocess.py::_score()`。
- **状态**：已实现，`"prob"` 可完全复现官方行为。
- **影响**：与原文的 loss 绝对值不可比，但分组是相对量，影响有限。

### D13　`iteration1` 按数据集取值

- **选择**：CIFAR-10 = 5、CIFAR-100 = 10（原文）；Tiny 待定（届时取 10）。
- **位置**：`configs/generated/*.json`；`tools/make_configs.py::DATASETS`。
- **状态**：已实现。

### D14　公共集使用确定性变换（无开关）

- **问题**：DS 的前提是"同一批样本、多个标注者投票"。
- **决定**：公共集只做 `ToTensor + Normalize`，**禁止**随机裁剪/翻转。
- **理由**：随机变换会让不同客户端看到不同的图，投票前提直接失效。
- **位置**：`stages/fedds_train.py::_public_tf`、`_public_predictions()`。
- **状态**：已实现，**故意不提供开关**。

### D15　学习率衰减默认关闭（已实现）

- **问题**：FedDS 原文用恒定 lr，但你的复现日志显示 100 轮时曲线仍在上升。
- **你的选择**：**按建议，主实验不加**。
- **位置**：`config.py::FedDSConfig.lr_decay = False`；`stages/fedds_train.py::_lr_for_round()`（余弦退火）。
- **状态**：**已实现**（v1.0 文档化时补的；此前只有字段没有逻辑，属于"假开关"，已修复）。
- **改法**：`--fedds.lr_decay=true`。

### D16　样本不会跨客户端出现（断言）

- **状态**：已实现，`manifest.validate()` 强制校验公共集与客户端互斥、并集覆盖全集。

---

## 4. 产物与工程类

### D17　预处理阶段把所有中间量都留下

- **你的选择**：全存。
- **内容**：每客户端的 LID 均值、本地 loss、proximal 系数 μ、权重哈希；每轮次的噪声客户端数、修正样本数、估计噪声均值、测试准确率。
- **位置**：`fedcorr/fedcorr_clients.jsonl`、`fedcorr_iterations.jsonl`。
- **理由**：论文要回答"识别准不准""改对多少"，没有这些数就只能凭结果猜。

### D18　每个 iteration 评估一次测试集

- **你的选择**：要。
- **位置**：`stages/fedcorr_preprocess.py::run()` 末尾 `evaluate(...)`。
- **说明**：官方是**每个客户端**都评估一次（19 小时里很大一部分花在这），我们改成每 iteration 一次，只影响日志，不影响算法。

### D19　断点粒度：iteration 级 + 每 20 个客户端

- **你的选择**：要。
- **位置**：`fedcorr_preprocess.py::_checkpoint()`、`fedds_train.py::_checkpoint()`。
- **注意**：会保存完整模型（ResNet18 约 45 MB/次、ResNet34 约 85 MB/次）。**磁盘紧张时**用 `runtime.checkpoint_models=false` 全部关掉。

### D20　修正后的标签另存快照，不覆盖 manifest

- **你的选择**：另存。
- **位置**：`fedcorr/corrected_labels.npy`；manifest 里的 `corrected_label` 字段保持初始值。
- **理由**：manifest 一旦被改写，就无法区分"原始噪声标签"和"修正后标签"，审计和复现都会断。

---

## 5. 环境与运行

### D21　GPU 与环境版本

- **你的选择**：同意我的建议。
- **GPU**：单卡 **RTX 4090 24 GB**（备选 3090）。理由：模型小、瓶颈在串行客户端训练，A100 性价比低。
- **环境**：Python 3.10 · PyTorch **2.5.1 (cu121)** · torchvision **0.20.1** · numpy **1.26.4** · scikit-learn **1.5.2**。

### D22　numpy 必须 < 2（硬约束）

- **证据**：官方 FedCorr 的 `lid_term` 里 `idx = np.ogrid[...]; idx[1] = ...`，numpy 2 返回 tuple，直接 `TypeError`。
- **影响**：任何照搬原版 FedCorr 的代码在 numpy 2 下都会崩。本工程自己的 LID 没用 `ogrid`，不受影响，但 `requirements.txt` 已锁 1.26.4。

### D23　`num_workers` 默认 0，服务器配置写 2

- **原因**：本地沙箱禁止命名管道，多进程 DataLoader 会 `PermissionError`。
- **位置**：`RuntimeConfig.num_workers = 0`；`configs/generated/*.json` 里是 2。

### D24　断点开关可整体关闭

- **位置**：`runtime.checkpoint_models`，默认 `true`。

### D25　现阶段只跑 idea 一格，对照后续再补

- **你的选择**："我现在不是要做对照时候，我就是要完成我的 idea 对应的编码，先看看效果好不好。"
- **位置**：`configs/idea_cifar10_k50.json` 与 `_smoke.json` 里 `arms` 只有 `["fedcorr+fedds"]`。
- **说明**：四臂实现都在（`fedavg` / `fedds` / `fedcorr+fedavg` / `fedcorr+fedds`），改成四臂只需改这一个字段。

---

## 6. 这些决定的已知影响（诚实清单）

| 决定 | 对结果的影响 | 是否可在论文里自圆其说 |
|---|---|---|
| D03 链式语义 | 与标准 FL 不同，收敛曲线形状不同 | ✅ 说明"沿用作者实现"即可 |
| D06 K=10 改 frac | 与论文超参不一致 | ✅ 说明"单标注者下 DS 无定义" |
| D07 不用预训练 | CIFAR-100 绝对值低于论文 | ✅ 说明"为保证两阶段可比" |
| D12 CE 在 logits | loss 尺度与原文不同 | ✅ 说明"修正作者实现的重复 softmax" |
| D14 公共集确定性 | 无负面影响 | ✅ 必须如此 |
| D15 关闭 lr 衰减 | 后期可能震荡 | ✅ 原文如此 |
| D25 只跑一格 | 暂时无法归因 | ⚠️ 正式论文必须补三臂 |

---

## 7. 想改某个决定，改哪里

| 想改 | 改什么 | 取值 |
|---|---|---|
| 客户端数 / 非 IID 程度 | `data.num_clients` / `data.dirichlet_alpha` | 10/50/100；0.5/10 |
| 标注者数量 | `fedds.frac` 或 `fedds.frac_by_num_clients` | 例如 `{10:0.5}` |
| 用回官方的 CE | `fedcorr.loss_on` | `"prob"` |
| 打分用确定性变换 | `fedcorr.train_transform` | `"eval"` |
| 开预训练（CIFAR-100） | `model.fedds_pretrained` | `true` |
| 开学习率衰减 | `fedds.lr_decay` | `true` |
| 关断点 | `runtime.checkpoint_models` | `false` |
| 跑四条对照臂 | `runtime.arms` | `["fedavg","fedds","fedcorr+fedavg","fedcorr+fedds"]` |
| 换数据集 | `data.dataset` + 重建配置 | `cifar10` / `cifar100` |

---

## 8. 附：验证过什么

| 验证项 | 方法 | 结果 |
|---|---|---|
| LID 与官方一致 | 加载官方 `util.py::lid_term` 对拍 | ✅ |
| ResNet18/34 与官方一致 | 同权重下单/前向输出对拍 | ✅ |
| 加权平均与官方一致 | 官方 `FedAvg` 对拍 | ✅ |
| DS 混淆矩阵与官方一致 | 官方 `dawid_skene_standard` 对拍 | ✅ |
| 单标注者退化 | 合成数据 | ✅ 可靠性恒为 1 |
| 数据守恒 / 注噪精度 | 单元测试 | ✅ |
| 四臂端到端 | 合成数据 | ✅ |
| **真实 CIFAR 精度** | — | ❌ 未验证（需服务器） |
| **GPU 耗时/显存** | — | ❌ 未验证 |

---

## 9. 后续新增（实现阶段的工程决定，非对话决策）

| 编号 | 内容 | 默认 | 开关 |
|---|---|---|---|
| D26 | FedCorr 结束后用干净标签做**诊断**（识别率、纠错精确率/召回率、估计 vs 真值相关性），只读、不回流训练 | 开 | 无（`analysis.py`） |
| D27 | **断点续跑**：每轮结束原子存模型+轮号，重启从下一轮继续；续跑结果与不中断逐位一致 | 关 | `runtime.resume` |
| D28 | **复用修正标签**：FedCorr 配置指纹一致时跳过 FedCorr，省几小时 | 开 | `runtime.reuse_correction` |
| D29 | 评估集/公共集张量**预变换缓存**，避免每轮重复 PIL 解码 | 开 | `runtime.cache_eval` |
| D30 | CUDA 不可用时**拒绝运行**（不静默降 CPU） | 拒绝 | `runtime.allow_cpu_fallback` |
| D31 | 全局随机数统一播种，每条实验臂独立派生种子 | 开 | `runtime.deterministic` |

上述默认值都写进 `configs/idea_*.json` 与 `configs/generated/*.json`。
