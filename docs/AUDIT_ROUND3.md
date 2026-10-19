# 第三轮代码审计（2026-10-18 晚）

> 触发原因：显卡暂无空闲，趁停机做一次更严格的静态 + 运行时审查
> 结果：**发现 1 个语义级严重错误、5 个会影响结果或可复现性的问题，全部修复；测试 26 → 29。**

---

## 一、语义级严重错误（已修）

### 1. FedDS 的客户端被串成了链

**问题代码**（`fedds_train.py::run`）：

```python
for client in selected:
    self.local.load_state_dict(self.model.state_dict())   # ① 从"全局模型"起步
    ...
    self.model.load_state_dict(states[-1])                # ② 但全局模型被覆盖成该客户端的权重
    predictions.append(self._public_predictions())
```

② 把全局模型覆盖了，于是下一个客户端的 ① 拿到的是**上一个客户端的模型**，而不是本轮全局模型。

**后果**：FedDS 退化成"客户端依次接力训练"（和 FedCorr 的链式一样），而不是联邦平均。精度曲线看起来正常，但语义完全错了，而且没法解释、没法写进论文。

**为什么之前的测试没抓到**：合成数据上精度本来就随机，四条臂的数字都差不多；没有测试检查"客户端起点"这个不变量。

**修复**：

```python
global_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
for client in selected:
    self.local.load_state_dict(global_state)        # 每轮所有客户端从同一份全局权重出发
    ...
    predictions.append(self._public_predictions(self.local))   # 用刚训好的本地模型做预测
# 聚合完再写回 self.model
```

**新增两个回归测试**（`tests/test_fedds_semantics.py`）：

- `test_clients_start_from_the_round_global_model`：记录每个客户端训练前的参数指纹，必须全部等于本轮全局模型；
- `test_global_model_is_not_mutated_while_collecting_predictions`：收集预测期间全局模型指纹不得变化。

---

## 二、影响结果与可复现性的问题（已修）

| # | 问题 | 后果 | 修复 |
|---|---|---|---|
| 2 | **模型初始化没有设种子** | 同一份配置跑两次，权重不同、结果不同；论文无法复现 | `seed_everything()`，FedCorr 与每条实验臂各自派生子种子；新增"同种子两次结果一致"测试 |
| 3 | **DataLoader 多进程下增强随机性未按 worker 分离** | 所有 worker 从同一 `random` 状态出发，裁剪/翻转序列重复；`num_workers>0` 时结果异常 | 接入 `worker_init_fn`，按 worker 重设 `random` / numpy 种子 |
| 4 | **日志以追加方式写** | 同一个实验名重跑时，新旧记录混在同一个 jsonl，读日志会看到重复轮次 | 每次 `run()` 开始时清空对应日志 |
| 5 | **GMM 遇到常数输入时会随意二分** | 若某一轮所有客户端 LID 相同，会被强行分出"噪声客户端"并触发错误修正 | 常数输入直接判定"全部干净"，`backend="constant"` |
| 6 | **命令行传 `arms` 会被拆成单字符** | `--runtime.arms=fedcorr+fedds` 变成 `('f','e','d',...)` | 统一用 `_as_list()` 支持逗号分隔字符串 |

---

## 三、顺手清理

- 删除 6 个文件里的 8 处未使用 import（静态检查脚本扫出来的）。
- 新增 `runtime.deterministic` 开关：默认开（`cudnn.deterministic=True`，可复现），追求速度可关。

---

## 四、这一轮的验证

| 验证 | 结果 |
|---|---|
| 全部单元测试 | **29 个全绿**（16 → 26 → 29） |
| CPU 端到端（K=10，2 次迭代，10 轮，四条臂） | 通过，产物完整 |
| 配置校验（27 份配置 + 命令行覆盖） | 全部通过 |
| 静态检查（未用 import / 语法） | 干净 |
| 客户端起点不变量 | 通过（新增） |
| 同种子可复现 | 通过（新增） |

---

## 五、仍未验证（必须靠服务器）

| 项 | 说明 |
|---|---|
| 真实 CIFAR 精度 | 本地只有 600 张伪造图的路径验证 |
| GPU 显存/耗时 | 无卡 |
| 200 轮收敛 | 无卡 |
| 多进程 DataLoader 在 Linux 上的实际行为 | 本地沙箱禁止命名管道，`num_workers=0` |

---

## 六、给第一次真跑的清单

1. 冒烟必须出现 `annotators=5`（K=50）且 `ds_iterations` 几十次内收敛；
2. `test_accuracy` 第一次迭代后应 > 10%；
3. 日志里不应出现 `CUDA requested but unavailable`；
4. 同配置跑两次，`summary.csv` 的数值应完全一致（可复现性自检）。
