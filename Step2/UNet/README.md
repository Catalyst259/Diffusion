# UNet：DDPM / DDIM 对比与 Turn3 条件训练

在 `Step2` 目录下执行，使用已有的 `ml-practice` 环境：

```powershell
conda activate ml-practice
python -m UNet.src.compare_samplers
python -u -m UNet.src.main
```

## 旧模型采样对比

`compare_samplers` 严格加载原有 `outputs/Turn2/weights/model_epoch10_step9300.pth`，不训练模型。默认生成 100 组对比，每组从同一个标准高斯噪声张量开始：

- DDPM：1000 次模型调用。
- DDIM：50 次模型调用，`eta=0`，确定性更新。
- 每组初始噪声种子为 `42 + group_index`（index 从 0 开始）；DDPM 的反向随机数种子另行记录。
- GIF 以相同扩散时间 t 对齐，包含共享初始噪声、DDPM、DDIM 三列。动画播放时长不是运行耗时。
- 采样保持原有数值规则，不裁剪预测的 x0；显示时固定裁剪到 `[0,1]`，与原训练归一化一致。
- 这是纯噪声生成测试，没有一一对应的真实目标图。原训练 preview 从训练图加噪后去噪，两者不能直接作为同条件画质比较。

结果位于 `outputs/DDIMContrast/`：100 个 `group_*.gif`、最终结果总览 `final_comparison.png`、未裁剪的 `final_samples.pt`、逐组记录 `results.csv` 和实验配置/耗时 `summary.json`。耗时包含轨迹采集，排除 GIF 编码；CUDA 测量前后同步并预热。使用 batch=20，单图耗时为批量摊销值。

已有结果目录不会被自动覆盖。新对比可用独立目录，例如：

```powershell
python -m UNet.src.compare_samplers --ddim-steps 100 --output UNet/outputs/DDIMContrast100
```

## Turn3_CFG：DiT 标签条件与标签丢弃

模型增加第三个输入 `c`，使用 MNIST 的整数标签 `0..9`：

```python
condition = time_mlp(t) + label_embed(c)
predicted_noise = model(xt, t, c)
```

时间和类别编码相加采用 DiT 的条件融合方式，融合结果经原有 ResBlock 内的投影注入；保留卷积、上下采样、GroupNorm 和跳跃连接。

按照 `DiT/official/models.py` 的 `LabelEmbedder`，条件模型默认 `class_dropout_prob=0.1`：训练时每个样本独立以 10% 概率把类别替换为可学习的空标签 `10`。标签表为 `11 × time_emb_dim`，最后一行代表无条件分支，与数字 `0` 不同；不会把标签向量简单置零。验证和采样调用 `model.eval()`，不随机丢弃标签。

Turn3_CFG 从头训练，保留原 Turn3 结果。其余配置保持：5 epoch、batch=64、Adam、lr=1e-4、seed=42、base_channels=96、time_emb_dim=256、1000 个扩散时间步、MNIST `ToTensor()` 的 `[0,1]` 范围。训练/验证均预测噪声，训练 preview 继续使用 DDPM 与当前训练 batch 第一张图片的随机 t，并传入其真实类别，`cfg_scale=1.0`。

训练 loss 包含随机丢弃标签的样本，测试 loss 使用完整真实标签，二者的条件不同。曲线的差距不能直接作为过拟合程度。

数据读取现有 `src/data/MNIST` 缓存，不受启动目录影响，不重新下载。

输出位于 `outputs/Turn3_CFG/`：

- 每 100 次更新保留一次权重和去噪 GIF，沿用原保存间隔与留档方式。
- `config.json`、每 epoch 的 `loss.csv` 和 `graph/loss_curve.png`。
- 训练结束保存 `weights/model_final.pth`，包含第 4690 步的模型；`completed.json` 标记完整训练结束。

新权重加载启用 10 类条件及标签丢弃。DDPM、DDIM 均支持 CFG：

```python
eps = eps_null + cfg_scale * (eps_cond - eps_null)
```

`cfg_scale=1` 是普通条件采样，只调用条件分支；`0` 是无条件预测；大于 `1` 增强条件引导，需要额外计算无条件分支。可从 `1、2、3` 对比画质与类别一致性，过强引导可能降低多样性。

```python
import torch
from UNet.src.model import UNet
from UNet.src.sample import inference

model = UNet(1, 1, time_emb_dim=256, base_channels=96,
             num_classes=10, class_dropout_prob=0.1).cuda()
model.load_state_dict(torch.load(
    "UNet/outputs/Turn3_CFG/weights/model_final.pth", map_location="cuda", weights_only=True,
))
labels = torch.arange(10, device="cuda")
noise = torch.randn(10, 1, 28, 28, device="cuda")
images = inference(noise, 1000, model, c=labels, sampler="ddim",
                   num_steps=50, cfg_scale=2.0)
```

权重沿用原来的纯 `state_dict` 格式，不包含优化器状态。加载旧无条件 Turn2 时保持 `num_classes=None`；加载旧 Turn3 的十行标签权重时须显式设置 `num_classes=10, class_dropout_prob=0.0`，且只支持 `cfg_scale=1`。不会为旧权重自动补一行未经训练的空标签。

本次训练结束后，`outputs/Turn3_CFG/cfg_preview.png` 展示两组固定初始噪声在标签 `0..9`、CFG `1/2/3` 下的 DDIM 50 步生成结果。各列是目标标签，各组内共享同一噪声；`cfg_samples.pt` 保存未裁剪结果，`verification.json` 记录检查和预览配置。可复现该预览：

```powershell
python UNet/outputs/Turn3_CFG/verify_preview.py
```

## 验证

```powershell
python -m unittest discover -s UNet/tests -v
```

测试覆盖 DDIM 独立公式、跳步边界、eta=0 确定性、eta=1 全步与 DDPM 一致、混合时间步的标签对齐、旧模型兼容、逐样本标签丢弃、空标签梯度、eval 行为、CFG 公式、训练/验证/preview 标签传递和 GIF 导出。
