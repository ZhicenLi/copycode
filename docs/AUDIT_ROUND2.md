# 第二轮代码审计（2026-10-18）

> 范围：`fusion/` 全部 24 个源文件 + 全部测试
> 方法：逐行阅读 → 找问题 → 修 → 跑测试 → 用官方源码对拍
> 结果：**发现 10 处问题，全部已修；测试从 16 个增加到 26 个，全绿。**

---

## 一、这轮改了什么

| # | 问题 | 严重度 | 表现 | 修复 |
|---|---|---|---|---|
| 1 | `lr_decay` 是假开关 | 中 | 配置里有字段，训练器从不读取 | 实现余弦退火 `_lr_for_round()`，默认仍关 |
| 2 | 评估集每次都重新做变换 | **高（性能）** | 200 轮 × 1 万张测试图 × 每轮 m 个客户端的公共集推理，全在重复 PIL 解码；实测会多花数小时 | 新增 `materialize()` + `make_cached_loader()`，评估/公共集张量只算一次；开关 `runtime.cache_eval` |
| 3 | CUDA 不可用时静默退回 CPU | **高（事故级）** | 200 轮在 CPU 上要跑几天，日志只有一行 warn | 默认**直接报错**，必须显式 `runtime.allow_cpu_fallback=true` |
| 4 | 公共集 loader 里带了 clean_label | 中（隐患） | DS 只用硬预测，暂未泄漏，但埋了雷 | 改成全零占位 |
| 5 | 缓存 manifest 不与数据集校验 | 中 | 换了数据根仍会复用旧划分，静默出错 | 新增 `_verify_manifest_labels()`，逐样本比对 clean_label |
| 6 | LID 内存 O(n²) | **高（K=10 会炸）** | K=10 时单客户端 4500 样本，稠密矩阵 + argsort 峰值约 500 MB | 改成分块计算，峰值降到 O(block×n)；**对拍官方函数仍逐位一致** |
| 7 | sklearn 缺失时静默用 numpy GMM | 中 | 本地与服务器分组可能不同却无提示 | 加 `RuntimeWarning` 明确告警；删掉死代码；补收敛判据 |
| 8 | `weighted_average` 无校验 | 中 | 权重长度不符或负数会静默算错 | 加长度断言 + 负权重截断 |
| 9 | 配置校验不全 | 中 | `frac>1`、`per_round>K`、空 arms、`local_ep<=0` 都能通过 | 全部补上，并加 4 个针对性测试 |
| 10 | `load_cifar` 写死 `download=True` | 低 | 数据已就位时仍会尝试联网 | 新增 `data.download` 配置 |

---

## 二、这轮新增的测试（16 → 26）

| 测试 | 覆盖 |
|---|---|
| `test_empty_arms_are_rejected` | 空实验臂 |
| `test_clients_per_round_cannot_exceed_clients` | 每轮客户端数超过总数 |
| `test_weighted_average_rejects_length_mismatch` | 权重长度不符 |
| `test_weighted_average_ignores_negative_weights` | 负权重 |
| 其余 22 个 | 数据守恒、注噪精度、DS 排序能力、单标注者退化、LID/ResNet/FedAvg/DS 与官方对拍、四臂端到端 |

---

## 三、真实数据路径验证

本地从 toronto 源下载 CIFAR-10 只有 **5 KB/s**（170 MB 要十小时），放弃。

改用**伪造的 CIFAR-10 格式数据集**（600 张训练 / 100 张测试、随机像素），让 torchvision 走完整真实路径：

- 真实 `CIFAR10` 类加载（pickle → `.data` / `.targets`）
- 真实 `RandomCrop + Flip + Normalize`
- 真实 ResNet18 + 6 层 CNN
- 完整 `FedCorr → 标签交接 → FedDS` 链路

结果：`600 train / public 61 / 5 clients`，FedCorr 修正 3 个标签，FedDS 跑完 3 轮并写出 `summary.csv`。

**这证明了数据管线、形状、变换、模型、交接都没问题；没证明的是真实数据上的精度和规模。**

---

## 四、仍然未验证的（诚实清单）

| 项 | 状态 | 说明 |
|---|---|---|
| 真实 CIFAR-10 精度 | ❌ | 只有服务器能跑 |
| GPU 显存/耗时 | ❌ | 同上 |
| 200 轮收敛性 | ❌ | 同上 |
| K=100 的 LID 耗时 | ⚠️ | 每客户端 450 个样本，分块后内存没问题，但 500 次调用的总时间要实测 |
| 断点恢复 | ⚠️ | 断点**只存不载**，没有 resume 逻辑；断点目前用于事后检查 |
| `_digest_state` 开销 | ⚠️ | 每客户端对 45 MB 参数做 SHA-256，500 次约 1–2 分钟，可接受 |
| 客户端断点间隔 | ⚠️ | 客户端级断点写死每 20 个，没有跟 `save_every` 联动 |

---

## 五、性能上的量级变化（按 200 轮估算）

| 项 | 修复前 | 修复后 |
|---|---|---|
| 测试集评估 | 200 轮 × 1 万张图重新做 PIL+Normalize | 只做一次，之后纯 GPU 前向 |
| 公共集推理 | 200 轮 × m 客户端 × 5000 张图重新做变换 | 只做一次 |
| LID 峰值内存（K=10） | ≈500 MB | ≈ 32 MB |
| 静默 CPU 回退 | 可能发生 | 直接报错 |

前两项按经验能省下 **1–3 小时/组**。
