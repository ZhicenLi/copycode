# AutoDL 使用说明（读官方文档后整理，针对本项目）

> 来源：AutoDL 帮助文档 <https://api.autodl.com/docs/>（快速开始、上传数据、公网网盘、公开数据、学术资源加速）
> 整理日期：2026-10-18

---

## 1. 开实例

1. 控制台 →「租用新实例」：选**计费方式 / 地区 / GPU 型号 / 数量 / 镜像**。
2. 镜像选**内置镜像**里带 `PyTorch 2.5.1 / Python 3.10 / CUDA 12.1` 的那个。
3. **状态显示「运行中」就开始计费**，不用时及时关机（关机后数据保留，但连续关机 15 天实例会被释放）。
4. 数据盘固定是 `/root/autodl-tmp`，**只有它容量大**；系统盘很小，别往 `/root` 下塞数据。

## 2. 传代码

| 方式 | 适用 | 说明 |
|---|---|---|
| **公网网盘**（官方强推） | 任意大小 | 实例里打开「AutoPanel → 公网网盘」，授权阿里云盘/夸克/OSS；速度 5–20 MB/s |
| JupyterLab 上传 | 小文件 | **只支持文件，不支持文件夹**——传 zip 再解压 |
| SCP | 文件夹 | 本地执行 `scp -rP <端口> 本地路径 root@region-x.autodl.com:/root/autodl-tmp` |
| FileZilla / XShell | 拖拉拽 | 图形化，适合传整个目录 |

本项目只有 **70 KB**，JupyterLab 直接上传 `fusion_delivery.zip` 即可。

## 3. 传数据（CIFAR-10）

**优先级从高到低：**

1. **AutoDL 公开数据**（最省事）：控制台 →「公开数据」→ 搜 `CIFAR`，复制实例内的路径 → 解压到 `/root/autodl-tmp`。官方文档：<https://api.autodl.com/docs/public_data/>
2. **公网网盘**：自己提前下载好再传。
3. **现场下载**：`torchvision` 会从 `cs.toronto.edu` 拉，**不在 AutoDL 学术加速范围内**，慢的话会很折磨（我本地实测 5 KB/s）。

如果数据已就位，把配置里的 `data.download` 设成 `false`，避免它再去联网。

## 4. 学术加速（只对 GitHub / HuggingFace 有效）

```bash
source /etc/network_turbo
# 用完建议关掉
unset http_proxy && unset https_proxy
```

官方明确只加速 `github.com`、`githubusercontent.com`、`githubassets.com`、`huggingface.co`。
**CIFAR 的下载源不在其中**，所以别指望它加速数据集。
文档：<https://api.autodl.com/docs/network_turbo/>

## 5. 装环境

镜像里通常已带 torch/torchvision。先验证：

```bash
python -c "import torch, torchvision, numpy, sklearn; \
print(torch.__version__, torch.cuda.is_available(), torchvision.__version__, numpy.__version__, sklearn.__version__)"
```

期望：`2.5.1 True 0.20.1 1.26.x 1.5.x`

缺什么补什么：

```bash
pip install scikit-learn==1.5.2
# 如果 numpy 被升到 2.x，降回来：官方 FedCorr 在 numpy 2 下会因 np.ogrid 崩溃
pip install "numpy<2"
```

## 6. 跑本项目

```bash
cd /root/autodl-tmp
unzip fusion_delivery.zip -d fusion && cd fusion

# 1) 冒烟（5–10 分钟）
python run.py configs/idea_cifar10_k50_smoke.json

# 2) 正式（约 6–12 小时）
python run.py configs/idea_cifar10_k50.json
```

如果数据不在 `/root/autodl-tmp/data`，加参数覆盖：

```bash
python run.py configs/idea_cifar10_k50.json --data.data_root=/你的/路径 --data.download=false
```

## 7. 跑之前必须确认的三件事

1. **日志里不能出现** `CUDA requested but unavailable`。现在代码会直接报错拒绝运行（除非显式允许 CPU 回退），这是故意的——200 轮跑 CPU 要几天。
2. **磁盘**：断点会存完整模型（ResNet18 约 45 MB/次）。紧张就在配置里设 `"checkpoint_models": false`。
3. **别用无卡模式跑训练**：无卡模式 CPU 极慢，只适合传文件、装环境。

## 8. 省钱的几个点

- 传数据、装环境可以先用**无卡模式**（便宜），确认无误再开卡跑。
- 跑之前先跑冒烟，别直接 200 轮。
- 训练脚本用 `nohup` 或 tmux 挂后台，避免 SSH 断线中断。
- 跑完立刻关机；**关机数据保留**，但连续关机 15 天实例释放。

## 9. 官方文档索引

- 快速开始：<https://api.autodl.com/docs/quick_start/>
- 上传数据：<https://api.autodl.com/docs/scp/>
- 公网网盘：<https://api.autodl.com/docs/netdisk/>
- 公开数据：<https://api.autodl.com/docs/public_data/>
- 学术加速：<https://api.autodl.com/docs/network_turbo/>
