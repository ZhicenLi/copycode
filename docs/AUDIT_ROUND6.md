# 第六轮：外部独立审计与修复（2026-10-18 深夜）

> 方式：开两个独立审计任务（一个看数据/配置层，一个看训练/流水线层），与我的自查交叉验证。
> 结果：外部审计共报 2 个 critical、9 个 major、约 20 个 minor；**已修 2 critical + 9 major + 主要 minor**。
> 测试：50 → **66 个**，全绿。

---

## 一、Critical

### X1. GPU 上第一次聚合就崩：权重张量在 CPU

`weighted_average()` 里：

```python
torch.tensor(weights).view(-1, ...)   # CPU 张量，(A,1) 不是 0 维
stacked * ...                          # stacked 在 cuda
```

PyTorch 只自动搬运 0 维 CPU 张量，`(A,1)` 会直接抛
`Expected all tensors to be on the same device`。

**本地全部测试都用 `device="cpu"`，所以完全看不见**；一旦上卡，第一轮训练完进入聚合就死。

修复：权重张量显式 `device=reference.device`，并把视图形状算一次复用。

### X2. `allow_cpu_fallback` 打印了却没改设备

原来只打印 "falling back to CPU"，`resolved_device` 仍是 `cuda:*`，后续 `.to(cuda)` 照样炸。
修复：该分支里真正执行 `resolved_device = torch.device("cpu")`。

---

## 二、Major

| 编号 | 问题 | 修复 |
|---|---|---|
| X3 | **断点续跑在真实重启后不等价**：续跑不会恢复 RNG，而 Dropout 与 torchvision 裁剪用的是全局随机流 | `_local_train` 开头按 (round, client) 派生种子重置 `torch/random/numpy`，使单次客户端更新成为纯函数；测试改为**新建 trainer 模拟进程重启** |
| X4 | 缓存 manifest 只校验 5 个字段，改 `noise_levels` / `min_client_samples` 会静默复用旧数据 | 校验补全，缺失字段直接报错 |
| X5 | `last_k=0` 时 `accuracies[-0:]` 取到全部，`last0_accuracy` 实际是全程均值 | 校验 `1 ≤ last_k ≤ rounds` |
| X6 | `iteration1=0` 静默变成"不纠错但标称 corrected" | 校验 `≥ 1` |
| X7 | `lid_k=0` 让 LID 全 0 → GMM 判所有客户端干净 → FedCorr 静默失效 | 校验 `≥ 2` |
| X8 | 未知顶层配置段（如 `"fedd"`）被静默忽略 | 未知段直接报错 |
| X9 | 空列表转成 float64 索引数组，极小数据集下 `IndexError` | 索引数组统一 `int64`，rest 为空时给明确错误 |
| X10 | `validate()` 的"公共集与客户端重叠"检查永远不可能触发（两边都从同一个数组派生） | 改成比较 `split=="public"` 与 `client_id == -1` 两个**独立定义**，并校验 `client_id` 范围 |
| X11 | manifest 校验和不覆盖 `clean_label` / `noise_rate` / 元数据；缺 meta 时直接跳过校验 | 校验和覆盖全部 7 个数组 + 协议字段；`load()` 缺 meta 直接报错并跑 `validate` |

---

## 三、Minor（已修）

- 配置整数字段接受浮点（`rounds=1.5`、`num_clients=10.0`）→ 统一类型校验。
- `min_client_samples` / `dirichlet_max_tries` / `public_bs` / `num_workers` 允许 0 或负数 → 加下界。
- `dirichlet_alpha` 接受 NaN/Inf；负数种子；`frac1=0` 触发 `ZeroDivisionError` → 统一守卫。
- `mixup_alpha=0`、`relabel_ratio=0`、`confidence_thres>1` → 加区间校验。
- `frac_by_num_clients` 值可为负 → 校验 `(0,1]`；且**有 FedDS 臂时要求每轮 ≥2 个标注者**。
- CLI 覆盖：`--data.noise_levels=0.5`（标量）、`--fedds.frac_by_num_clients=10:0.3,...`（列表字符串）、`--runtime.exp_name=123`（整数）→ 全部正确解析。
- 三个**从未被读取**的配置项（`fedcorr.clean_set_thres`、`fedds.normalize`、`runtime.log_every`）→ 删除，避免"假开关"。
- `derive_seed` 的分隔符拼接会碰撞（`(1,"2|3")` 与 `(1,2,3)`；`2` 与 `"2"`）→ 改为带类型与长度前缀。
- `evaluate()` 在空 loader 上返回 0.0 → 改为抛错。
- `IndexedImages(transform=None)` 会喂 uint8 HWC 给模型 → 直接拒绝。
- FedCorr 的中间断点写死每 20 个客户端，且旧断点从不清理（每组累积约 0.7–1.1 GB）→ 改用 `save_every`，启动时清理旧断点。
- 断点命名 `i{iteration}_s{step+1}` 与 `i{iteration+1}_s0` 语义不一致 → 统一为「已完成迭代」计数。

---

## 四、外部审计确认无问题的部分

- 索引空间：客户端局部下标与全局 `sample_id` 全程一致，未发现 off-by-one。
- 语义：FedCorr 的链式、FedDS 的"每轮同一全局快照 + 聚合前不改全局模型"均正确。
- 交接：只有标签跨阶段，`manifest.noisy_label` 从未被改写，`arm_labels.copy()` 防止跨臂污染。
- 断点写序：先写日志、后原子写断点，恢复时按断点截断日志，不会重复计数。
- 复杂度：LID 分块 O(block·n)，未发现意外的 O(n²) 内存。

---

## 五、验证

| 项 | 结果 |
|---|---|
| 单元测试 | **66 个全绿**（新增 16 个校验回归测试） |
| 28 份配置 | 全部通过更严格的校验 |
| 端到端（CLI，四臂） | 通过，产物齐全 |
| 断点续跑（模拟进程重启） | 与不中断逐位一致 |
| 臂间共享初始化 | 通过 |

---

## 六、教训

这轮 2 个 critical 都**只在 GPU 或跨进程时暴露**，本地 CPU 测试永远看不见。说明"跑通了测试"不等于"能在目标环境跑起来"——这也是为什么第一次上卡要先冒烟，而不是直接 200 轮。
