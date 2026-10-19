# 代码审计：与 FedCorr / FedDS 原实现的逐项对照

> 审计对象：`fusion/`（本工程）
> 对照基准：`code/fedcorr`（官方 `d9c0d86`）、`code/fedds_official`（官方 `cfb26b2`）、`code/fedds_repro`（修复版 `7b97ea5`）
> 审计方式：逐函数对照 + 在真实 torchvision 路径下重跑全部测试
> 结论先行：**算法主干与原文一致（31 项严格对齐、8 项等价实现）；有 3 处有意偏离、1 处原代码缺陷修正、3 处本工程问题（已修）；数值未在真实数据上验证。**

---

## 0. 可信度分级

| 级别 | 含义 | 数量 |
|---|---|---|
| **A 严格对齐** | 语义与实现都能一一对应到原代码行 | 31 |
| **B 等价实现** | 写法不同（向量化、PIL、log 空间），数学等价 | 8 |
| **C 有意偏离** | 为满足本研究的协议或修正明显缺陷 | 3 |
| **D 新增** | 原论文与原代码都没有的部分 | 1 + 9（见第 3 节） |

---

## 1. FedCorr 预处理逐项对照

原文件：`code/fedcorr/main.py`、`util/local_training.py`、`util/util.py`、`util/dataset.py`

| # | 环节 | 原实现 | 本实现 | 级别 | 说明 |
|---|---|---|---|---|---|
| 1 | 外层迭代 | `main.py:86` `for iteration in range(iteration1)` | `fedcorr_preprocess.py: run()` 同名循环 | A | |
| 2 | 客户端遍历 | `main.py:100-101` 循环 `int(1/frac1)` 次，每次 `choice(K, int(K*frac1))`，用 `prob` 置零实现不放回 | `rng.permutation(K)` 打乱后逐个访问 | **B** | 集合相同（每个 iteration 访问全部客户端一次），但**随机流不同**，结果不会与原文逐位一致 |
| 3 | 每步聚合 | `main.py:131-134` `FedAvg(w_locals, dict_len)`，`w_locals` 只有 1 个元素 → 直接返回该客户端权重 | 每个客户端训练完直接 `model.load_state_dict(local)` | **B** | 数值完全等价（单元素加权平均即其本身） |
| 4 | 本地优化器 | `local_training.py:64` `SGD(lr, momentum)` | 同名同参 | A | |
| 5 | 本地 epoch/batch | `args.local_ep`、`args.local_bs` | `cfg.fedcorr.local_ep/local_bs` | A | 默认值同原文（5 / 100） |
| 6 | mixup | `local_training.py:75-79` `np.random.beta` + `torch.randperm`，用 `lam*CE(a)+(1-lam)*CE(b)` | 同公式 | **B** | 我用**每客户端独立 RNG**，原代码用全局 numpy 随机流。语义相同、可复现性更好 |
| 7 | proximal 项 | `local_training.py:86-92`：`batch_idx>0` 时加 `beta*mu*sqrt(Σ‖w−w_g‖²)` | 同式，锚点为训练前权重 | A | |
| 8 | proximal 系数 `mu` | `main.py:93-96`：第 0 轮全 0，之后用上一轮的 `estimated_noisy_level` | `estimated_noise[client] if iteration>0 else 0` | A | |
| 9 | 打分用的模型 | `main.py:125` 用 `net_local`（训练后） | 用 `self.model`（已载入该客户端权重） | A | 单客户端聚合后两者权重相同 |
| 10 | 打分用的数据集 | `dataset_train`，带**训练期随机变换**（`RandomCrop`+`Flip`+`Normalize`） | `_score_tf`，默认 `train` | A | 可用 `fedcorr.train_transform="eval"` 切换 |
| 11 | 前向与 loss | `util.py:31-55`：先 `softmax`，再把概率喂给 `CrossEntropyLoss` | 默认在 **logits** 上算 CE | **C** | 原代码等于套了两层 softmax（`softmax(softmax(logits))`），loss 被压扁。`loss_on="prob"` 可切回原文 |
| 12 | LID 输入 | `main.py:125-126` `get_output` 的 **softmax 概率向量** | `_score` 返回的 probs | A | 注意：LID 不是算在特征上，是算在预测分布上 |
| 13 | LID 计算 | `util.py:58-74`：`cdist` 取 k 近邻，`-k/(Σlog(v/v_last)+eps)` | `core/lid.py` 同式，`cdist` 换成广播 | **B** | 默认 k=20，与原文一致（原代码没传 `args.LID_k`，用的是函数默认 20） |
| 14 | 客户端 LID | `main.py:129` `LID_client[idx] = mean(LID_local)` | 同 | A | |
| 15 | LID 累积 | `main.py:136` 跨 iteration 累加 | `lid_accumulated += lid_this_iteration` | A | |
| 16 | 客户端分组 | `main.py:140-146` 对累计 LID 做两分量 GMM，均值小的为 clean | `split_two_components` + `noisy_mask` | A | 服务器有 sklearn → 与原文同一个实现 |
| 17 | 噪声客户端才估噪声 | `main.py:150` `for client_id in noisy_set` | 同 | A | |
| 18 | 样本级噪声估计 | `main.py:153-159` 对该客户端 loss 做两分量 GMM，噪声比例 = 高均值分量占比 | 同 | A | 注意：原代码的 `loss_accumulative_whole` 每个 iteration 清零，**并不跨迭代累积**，名字有误导；我保持一致 |
| 19 | 修正预算 | `main.py:169` `int(n * est_noise * relabel_ratio)` | 同式 | A | `relabel_ratio` 默认 0.5 |
| 20 | 候选筛选 | `main.py:169` 按 loss 降序取前 budget 个 | `np.argsort(-loss)[:budget]` | A | |
| 21 | 置信度过滤 | `main.py:170` `prob.max(1) > confidence_thres` | 同 | A | 默认 0.5 |
| 22 | 标签更新 | `main.py:173-175` 用全局模型 `argmax` 覆盖 | 同 | A | |
| 23 | 修正时机 | 每个 iteration 末尾一次 | 同 | A | 识别与修正交替，不是分两段 |
| 24 | 全局模型在修正时的状态 | 是**最后一个被访问客户端**的模型 | 同 | A | 这是链式语义的直接后果 |
| 25 | 每客户端评估 | `main.py:121-122` 每个客户端都跑一次测试集 | 只在每个 iteration 末尾评估一次 | **D** | 原代码这一步是纯诊断，占掉大量时间；去掉不影响算法，只影响日志 |
| 26 | stage 2/3 | `main.py:181+` | 不执行 | **C** | 按计划书，只借 stage-1 |

---

## 2. FedDS 逐项对照

原文件：`code/fedds_official/main.py`、`dawid skene.py`

| # | 环节 | 原实现 | 本实现 | 级别 | 说明 |
|---|---|---|---|---|---|
| 1 | 客户端选择 | `main.py:71-72` `m=max(int(frac*K),1)`，`np.random.choice(K,m,replace=False)` | `clients_per_round()` 同式 | A | |
| 2 | K=10 的 frac | 原文未定义，按 0.1 反推则 m=1 | **K=10 时自动用 0.3（m=3）** | **C** | 1 个标注者时 DS 数学上失效，见 `tests/test_core.py::test_single_annotator_carries_no_information` |
| 3 | 本地训练 | `main.py:14-25` `SGD(lr,momentum)` + `CrossEntropyLoss` + `local_ep` | 同 | A | |
| 4 | 模型初始化 | `main.py:62` `CNNCifar(args)`，每轮每个客户端 `deepcopy(net_glob)` | 同 | A | |
| 5 | 公共集预测 | `main.py:79-89` 新建 client 模型载入本地权重，`torch.max` 取硬预测 | `predict_labels` | A | |
| 6 | 预测矩阵方向 | `main.py:91` `list(zip(*client_predictions))` → (样本, 客户端) | `np.stack(predictions, axis=1)` | **B** | 方向一致，写法不同 |
| 7 | DS 调用 | `main.py:102` `ds_func(matrix, num_classes)` | `dawid_skene(matrix, ...)` | A | |
| 8 | DS E 步 | `dawid skene.py:46-57` 连乘各标注者概率 | log 空间求和 | **B** | 数学等价；log 空间避免标注者变多后下溢 |
| 9 | DS M 步 | `dawid skene.py:59-68` 逐样本逐类累加再行归一 | `posterior.T @ one_hot` 矩阵乘 | **B** | 等价且快很多 |
| 10 | 类先验 | 原实现**从不更新** `p_c`（与论文 Eq.9 不符） | 默认不更新，`ds_use_class_prior` 可开 | A | 默认与作者实现一致 |
| 11 | 收敛判据 | `max|Δposterior| < 1e-5`，`max_iter=500` | 同 | A | 额外返回迭代次数与是否收敛（原文不暴露） |
| 12 | 可靠性 | `main.py:106` `mean(diag(confusion))` | 同 | A | |
| 13 | 聚合 | `FedAvg(w_locals, weights=reliability)`，权重归一化 | `weighted_average` 同 | A | |
| 14 | 整数 buffer | 原版直接平均，`load_state_dict` 时隐式截断为 int64 | 显式四舍五入后转回 int64 | **B** | BN 的 `num_batches_tracked`；差异仅在这个计数器 |
| 15 | FedAvg 基线权重 | `main.py:94` 用客户端样本量 | 同 | A | |
| 16 | 每轮评估 | `main.py:114` 测试集准确率 | 同 | A | |
| 17 | 汇总口径 | `main.py:128-132` 最后 10 轮均值 | `last_k=10` 同 | A | |

---

## 3. 本工程新增（原文没有）

| # | 内容 | 为什么必须有 |
|---|---|---|
| 1 | 数据协议：先切 10% 公共集 → Dirichlet → 注噪 | 计划书的核心设定，两篇论文都没有这个组合 |
| 2 | `manifest` 契约（sample_id/client_id/三种标签/哈希） | 标签交接的前提；保证两阶段看到同一批样本 |
| 3 | 四条实验臂共用一份修正标签 | 让 2×2 对照成立，且 FedCorr 只跑一次 |
| 4 | 标签快照（不覆盖 manifest） | 审计需求：原始数据不动 |
| 5 | 每客户端 LID / 噪声估计 / 修正数记录 | 论文里"识别准不准""改对多少"要用 |
| 6 | 每轮 DS 可靠性、迭代数、收敛标志、权重最大值 | 诊断 DS 是否真的在工作 |
| 7 | 种子派生体系 | 换一个流不会污染另一个 |
| 8 | 配置 + 命令行覆盖 + 24 份生成配置 | 你要跑多组对比 |
| 9 | 断点（iteration 级 + 每 20 客户端） | 13 小时一组，断了不能重来 |

---

## 4. 有意偏离清单（论文里必须写）

| # | 偏离 | 原因 | 影响 | 开关 |
|---|---|---|---|---|
| 1 | CE 算在 logits 上 | 原代码在概率上再算 CE，等于双层 softmax | loss 尺度不同，GMM 分组点不同 | `fedcorr.loss_on="prob"` 回退 |
| 2 | 只跑 stage-1 | 计划书要求 | 无 stage-2 后的二次修正 | 无（设计） |
| 3 | K=10 用 frac=0.3 | 1 个标注者时 DS 无信息 | 与论文 K=10 不可直接比 | `fedds.frac_by_num_clients` |
| 4 | FedDS 阶段默认从头训练 | 与 FedCorr 阶段可比 | CIFAR-100 绝对值低于原文 | `model.fedds_pretrained` |
| 5 | 公共集确定性变换 | DS 要求同一批输入 | 无（这是正确做法） | 无 |
| 6 | 客户端遍历用 permutation | 复现性更好 | 与原文随机流不同 | 无 |

---

## 5. 审计中发现并已修复的缺陷

### 5.1 【严重】torchvision 变换的 HWC/CHW 错位

`IndexedImages` 把 uint8 的 `(H, W, C)` 张量直接交给 torchvision。`ToTensor` 对张量原样返回，于是 `RandomCrop` 把 `C` 当成通道维、在错误的两维上裁剪。

- **本地为什么没炸**：本地没装 torchvision，走的是 numpy 兜底分支。
- **装了 torchvision 后**：`ValueError: Required crop size (32, 32) is larger than input image size (40, 11)`，在你服务器上必崩。
- **修复**：`__getitem__` 先转 PIL 再交给变换；两条分支现在都从 PIL 出发。
- **验证**：`torch.stack([ds[i][0] ...]).shape == (4, 3, 32, 32)`，模型前向通过。

### 5.2 mixup 使用全局随机流

原 `_local_train` 用 `np.random.beta` / 全局 `torch.randperm`，导致同一种子下换客户端顺序就会变。已改为每客户端派生独立 RNG（`derive_seed(seed, "mixup")`）。

### 5.3 `num_workers>0` 在本机不可用

沙箱禁止命名管道，DataLoader 多进程会 `PermissionError`。已把默认值改为 0，服务器配置里显式设 2。

---

## 6. 仍未验证 / 存疑（诚实清单）

| # | 事项 | 风险 | 怎么验证 |
|---|---|---|---|
| 1 | **数值正确性未在真实数据上验证** | 高 | 服务器跑一组 K=50 CIFAR-10，看 FedCorr 修正率与最终精度是否落在合理区间 |
| 2 | GMM 后端差异：本地 numpy 兜底、服务器 sklearn | 中 | 服务器上确认日志里 `gmm_backend="sklearn"`；两种实现在双峰数据上一致（已有测试） |
| 3 | 修正预算 `int(n·ρ̂·0.5)` 的边界 | 低 | 小客户端（n<10）时 budget 可能为 0，已跳过 |
| 4 | 原代码 `get_output` 的 cudnn 确定性设置我没有复刻 | 低 | 只影响 GPU 上的可复现性 |
| 5 | 归一化参数：FedCorr 用 ImageNet 均值，FedDS 官方用 CIFAR 均值 | 中 | 我统一用 spec 里的 FedCorr 参数；两阶段一致，但与 FedDS 原文不同 |
| 6 | 多标签/多任务未支持 | 无 | 当前只做单标签分类，与计划书一致 |
| 7 | 断点只存不载（没有 resume 逻辑） | 中 | 目前断点用于事后检查；续跑还没实现 |
| 8 | 与原文"逐位一致"**不成立** | 中 | 随机流不同，只能保证"同分布"，不能保证"同数值" |

---

## 7. 你可以自己复核的四个点

```bash
# 1) DS 在单标注者时确实无信息（退化的证据）
python -m unittest tests.test_core.TestDawidSkene -v

# 2) 数据守恒与注噪精度
python -m unittest tests.test_core.TestManifest -v

# 3) 四条臂端到端（合成数据，约 15 秒）
python -m unittest tests.test_pipeline -v

# 4) 完整套件
python -m unittest discover -s tests -v
```

---

## 8. 我的判断

- **算法骨架可信**：FedCorr 的"链式预处理 + 累计 LID + GMM + 置信度修正"和 FedDS 的"公共集预测 + DS + 加权聚合"都能逐行对上原实现。
- **不能承诺"跑出来就是论文的数"**：随机流、数据协议、CE 的计算位置都不同，绝对值会有偏差。
- **最大的未知不是代码，是数据**：真实 CIFAR 上 FedCorr 的修正率、DS 的可靠性分布是否正常，只能跑一次才知道。

建议第一次在服务器上只跑 **K=50、α=0.5、local_ep=5、rounds=20**（约 1 小时），先看四个数：修正率、噪声客户端数（应接近 60%×50=30）、DS 迭代次数（应收敛，几十次内）、最终精度（应明显高于随机 10%）。
