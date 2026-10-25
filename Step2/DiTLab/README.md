# DiTLab：DiT + Drift / EDM + MNIST

## 与 UNet 参数量匹配的两组五轮实验

`outputs/dit57m_compare/` 使用同一 DiT：patch 4、hidden 512、depth 12、heads 8，
每个模型 **57,656,848** 个参数，与此前 UNet 的 56,960,545 相差 +1.22%。
Drifting 与 EDM 分别从相同随机初始化独立训练 5 epochs，各 4690 次更新。
保持完整 MNIST 划分、batch 64、AdamW 学习率 `1e-4`、weight decay 0、
梯度裁剪 2、seed 42、FP32；训练数据顺序相同，预览使用相同类别及初始噪声。
EDM 沿用修正后的 `p_mean=0`、`p_std=1.2`；预览 CFG 2、32 步 Heun。
Drifting 沿用每类 8 个生成粒子及原温度设置，预览为单次前向。

在 `Step2` 目录使用 `ml-practice` 环境运行：

```powershell
python -m DiTLab.run_capacity_comparison --stage preflight
python -m DiTLab.run_capacity_comparison --stage train
python -m DiTLab.compare_methods
```

预检在独立进程内验证完整优化更新、显存及采样，随后丢弃预检权重。
正式训练重新初始化，并核对两组初始权重 SHA-256。两组依次占用 GPU。
若训练中断，运行 `python -m DiTLab.run_capacity_comparison --stage train --resume`；
会跳过已完成组，从未完成组的最新 checkpoint 恢复。
另开一组时，三个命令都传入同一个新的 `--root` 路径。

每组在 `drift/` 或 `edm/` 下保存与 `outputs/default/` 一致的配置、逐轮 loss、
曲线、checkpoint 及预览，另有 `train.log` 和 `runtime.json`。
训练完成后，`compare_methods` 核验完成状态、参数量、配对配置和预览，输出：

- `drift_loss.png`：Drifting 的逐轮训练/测试 loss。
- `edm_loss.png`：EDM 的逐轮训练/测试 loss。
- `drift_vs_edm.png`：Test / Drifting / EDM 三排配对抽样图。
- `comparison.json`：配置、原始 loss、完成步数和产物校验信息。

曲线使用真实逐轮均值，不平滑；Drifting 的归一化目标不保证单调下降。
两种目标的 loss 数值不可直接比较，相同参数量与更新次数也不等于相同计算量。

本组于 2026-10-26 完成，两个模型均从头训练，未从小模型权重续训。
初始权重 SHA-256 一致，两组各 10 份预览的 seed、测试索引和标签全部匹配；
最终 checkpoint 均为第 5 轮 / 第 4690 步，参数和 loss 均通过有限值检查。

| 方法 | Train loss：第 1 → 第 5 轮 | Test loss：第 1 → 第 5 轮 | 训练及产物耗时 |
| --- | --- | --- | --- |
| Drifting | 8.611611 → 8.472469 | 8.685057 → 8.677141 | 971.66 秒（16.19 分钟） |
| EDM，p_mean=0 | 0.542948 → 0.292195 | 0.414758 → 0.284861 | 659.36 秒（10.99 分钟） |

最终固定预览中，EDM 数字笔画较清晰，部分形状仍有变形；Drifting 能生成对应类别，
但笔画明显更模糊，同类样本形态也较接近。此处是这组配对预览的观察，未据此计算总体质量指标。
三张汇总图均已检查，完整数值、GPU 显存及运行信息见 `outputs/dit57m_compare/comparison.json`。

同卡推理测速（`python -m DiTLab.benchmark_inference`）：两个最终模型均常驻 GPU，
每种 batch 下分别预热 5 次，再交替先后顺序各测 30 次。使用 `torch.inference_mode()`、
网络 FP32、EDM 原有 FP64 积分，无 autocast 或 compile；CFG 2、EDM 32 步 Heun。
每次通过 CUDA 同步包围完整 sampler，计入 CPU 调度及采样开销，排除模型加载、
输入生成/搬运、预热和图片保存。结果为当前机器热态实际生成延迟：

| 批量大小 | Drifting 中位整批耗时 | EDM 中位整批耗时 | Drifting 加速倍数 |
| --- | --- | --- | --- |
| 1 张 | 11.56 ms | 689.93 ms | 59.67× |
| 10 张 | 10.26 ms | 818.78 ms | 79.78× |

batch 10 的摊销耗时分别为 1.03 / 81.88 ms 每张，不等于单张请求延迟。
批量大小会改变 GPU 利用率；小批次还受 CPU 调度和频率波动影响。
完整单次测量、分位数、吞吐及软件配置保存在 `outputs/dit57m_compare/inference_benchmark.json`。

## EDM 去噪扩散对照实验

在 `Step2` 目录运行（使用包含 PyTorch 的 `ml-practice` 环境）：

```powershell
python -m DiTLab.sample --checkpoint DiTLab/outputs/edm_cfg_balanced/weights/latest.pt
python -m DiTLab.train --config DiTLab/configs/mnist_edm_cfg.json --output-dir DiTLab/outputs/edm_cfg_balanced_repeat
```

该配置以 `outputs/default/config.json` 中实际完成的 Drifting 实验为对照：
MNIST 官方 60,000/10,000 划分、28×28 灰度像素空间、原始数字标签、
DiT patch 4 / hidden 192 / depth 4 / heads 6、batch 64、AdamW 学习率 `1e-4`、
weight decay 0、梯度裁剪 2、seed 42、5 epochs（4690 次更新）。
两者使用同一模型类、相同参数量及初始化；不使用 VAE、额外预训练网络或 EMA。
当前推荐结果位于 `outputs/edm_cfg_balanced/`，保留相同的 `config.json`、`losses.csv`、
`graph/loss_curve.png`、`weights/latest.pt` 和 10 列双排预览及 JSON。
这次诊断训练每轮保存预览（938 次更新）；新运行配置仍为每 500 次更新。
预览种子、测试集索引和初始高斯噪声与原实验同一步数一致。

基线使用 [EDM](https://arxiv.org/abs/2206.00364) 的预条件去噪目标：
`sigma_data=0.5`、`log(sigma) ~ Normal(0, 1.2²)`，噪声等级嵌入为 `log(sigma)/4`。
原实验沿用 EDM 的 `p_mean=-1.2`，诊断后针对本小型 DiT 的五轮预算调整为 `0`，见下文。
DiT 原有的标量嵌入在此用于噪声等级；训练时以 0.1 的概率将类别嵌入置零，
不新增参数，数字 0 仍是正常类别。测试 loss 使用完整测试集、固定噪声随机流和真实标签，
不丢弃标签。采样采用 Karras 调度（`rho=7`、sigma 从 `80` 至 `0.002`）、
32 个非零噪声等级的确定性 Heun 求解器，最后 Euler 更新至零，CFG 强度为 2。
训练和采样的公式参考 [NVlabs 官方实现](https://github.com/NVlabs/edm)。

Latent Diffusion 是在 VAE 编码的潜空间中进行扩散，不代表某种唯一最优的去噪算法。
这里选择成熟的 EDM 作为保持相同 DiT 和像素表示的基线；已做有限单变量诊断，未做算法排名或全面调参。
EDM 的 CFG 为条件/无条件预测的外推，Drifting 的 CFG 则由生成器直接学习；
虽然预览强度同为 2，两者语义和计算量不同。32 步 Heun 需要 63 次去噪器调用，
每次将两个 CFG 分支合并成一个 batch，因此每张图为 126 次等效网络评估；Drifting 为 1 次。
相同 epoch/更新次数不等于相同 FLOPs，且两种训练目标的 **loss 数值不能直接比较**。
这是单种子短训对照，预览和各自 loss 趋势可用于初步观察，不能据此断言最终算法优劣。

续训仍使用原配置与优化器状态：

```powershell
python -m DiTLab.train --resume DiTLab/outputs/edm_cfg_balanced/weights/latest.pt --epochs 10
```

原始 `outputs/edm_cfg/` 对照于 2026-10-25 在 RTX 5070 Laptop GPU 上完成，训练及定期产物共耗时
285.31 秒。EDM 首轮 train/test loss 为 `0.467783 / 0.318943`，
第 5 轮为 `0.260958 / 0.259566`。最终固定预览仍有明显碎笔画和标签不匹配，
这组短训预算下的数字成形效果弱于原 Drifting 预览；不能由此推断充分训练后的优劣。
`outputs/edm_cfg/graph/drift_vs_edm.png` 为相同最终步数的并排图；
`summary.json`、`runtime.json`、`train.log` 保存核验摘要、耗时和完整日志。
原实验权重/配置/loss 的 SHA-256 已确认未变，10 组预览的种子、索引、标签全部一致。
原 42 项测试及新增 9 项 EDM 测试通过，最终 checkpoint 重新加载采样也通过。

### 2026-10-25：碎片生成的诊断与修正

原模型能够去除低噪声，但高噪声下没有充分学会整体数字和类别约束。
固定输入复现、独立公式测试、采样消融和 CPU 小样本过拟合排除了输出拼接、模型无法学习、
CFG 公式和采样步数作为主要原因。UNet 的确能从纯噪声生成数字，但其参数量为 56,960,545，
是此 DiT（2,834,512）的约 20.1 倍，且有多尺度卷积与 skip，不能只按 epoch 推断相同训练难度。

只将训练 `p_mean=-1.2` 改为 `0`，保持 DiT、参数初始化、数据、学习率 `1e-4`、
batch 64、seed 42、5 epochs / 4690 更新及采样配置不变。
`sigma>2` 的训练样本占比由 5.73% 增至 28.18%，增加了需要学习全局结构与类别先验的训练任务。
该单变量干预显著改善生成；因此原先噪声分配不适合本次小 DiT 的短训预算，是已获得实验证据的关键因素。
这不是 EDM 公式错误，也不说明这一分布对所有模型最优。

固定种子 873、每组 100 张生成图的诊断结果：

| 干预 | 总 epochs | 分类器标签一致率 |
| --- | ---: | ---: |
| 原配置 | 5 | 7% |
| 原权重，仅 CFG 2→1 | 5 | 7% |
| 原权重，仅 Heun 32→128 步 | 5 | 7% |
| 原权重，仅 sigma_max 80→10 | 5 | 8% |
| 训练及采样噪声编码同时 ×100 | 5 | 10% |
| 仅训练 p_mean −1.2→0 | 5 | 70% |

另外进行了四组一轮筛查：原设置 9%、编码 ×100 为 11%、p_mean=0 为 10%、学习率 `5e-4` 为 9%。
一轮结果不充分，因此继续了噪声编码与噪声分布两个假设至五轮；未把一轮学习率实验当作完全排除优化器因素的证据。

最后使用**未参与上述筛查的 seed 19873、500 张相同新噪声**复核：原模型 11.2%，修正后 76.4%。
分类评估器只用 MNIST 训练集的固定 50k/10k 划分训练/验证，验证准确率 98.48%；未用官方测试集训练它。
这是标签一致性的诊断代理指标，不是 FID 或视觉质量保证；修正后仍有错类、碎笔画和单种子局限。
噪声分布改变了 loss 平均所对应的任务，原始与修正后的 loss 大小也不能直接比较。

产物：

- `outputs/edm_cfg_balanced/graph/fresh_noise_comparison.png`：新噪声前 30 张，不筛图。
- `outputs/edm_cfg_balanced/graph/before_after.png`：同最终 step 的原始/修正预览。
- `outputs/edm_cfg_balanced/diagnosis.json`：结论和核验数值。
- `outputs/edm_cfg_balanced/provenance.json`：诊断运行来源；复制结果时仅更正 checkpoint 输出路径。
- `outputs/edm_diagnosis/`：全部消融、独立评估器和过拟合检查；原 `edm_cfg`、`default` 保留。
- `configs/mnist_edm_cfg_legacy.json`：原始配置；`mnist_edm_cfg.json` 现在采用修正配置。

53 项单元/集成测试通过。生成回归检查在原 checkpoint 上失败，在修正 checkpoint 上通过（阈值 50%）。
可重新运行：

```powershell
python -m DiTLab.diagnostics.check_generation --checkpoint DiTLab/outputs/edm_cfg_balanced/weights/latest.pt --output DiTLab/outputs/edm_diagnosis/recheck
```

以下为原 Drifting 实验说明。

独立实验目录，原有 `UNet`、`DiT`、`Drift` 等目录不做修改。默认使用 MNIST 官方训练集（60,000 张）和测试集（10,000 张），原生数字标签 `0–9` 作为类别条件。

## 运行

在 `Step2` 目录运行：

```powershell
conda activate ml-practice
python -m DiTLab.train
```

也可进入 `DiTLab` 后运行 `python train.py`。沿用现有环境里的 PyTorch、torchvision、Pillow、matplotlib，不需要 JAX、timm、VAE 或预训练特征提取器。首次运行将 MNIST 下载到 `DiTLab/data`；如已有本地数据，可只读复用：

```powershell
python -m DiTLab.train --data-dir ../data --no-download
```

默认配置：

| 参数 | 默认值 |
| --- | --- |
| Epoch | 20（沿用当前 `config.py`） |
| Batch size | 64 |
| 优化器 | AdamW，学习率 `2e-5`，weight decay `0` |
| 模型 | DiT，28×28 单通道，patch 4，hidden 192，4 层，6 个 attention heads |
| Drift | 像素空间；每个 batch 中出现的类别生成 8 个样本；温度 `0.02, 0.05, 0.2` |
| CFG | 训练强度在 `[1,4]` 对数均匀采样；预览默认 `2.0` |
| Preview | 每 500 次 optimizer 更新，从官方测试集随机无放回抽 10 张 |
| 权重保存 | 每 500 次更新覆盖 `weights/latest.pt`；每个 epoch 完成和训练结束时也保存 |
| Loss | 每个 epoch 的训练集平均 loss、完整测试集平均 loss，均按真实样本数加权 |
| 设备 | 有 CUDA 时自动使用 CUDA，否则 CPU |

这里的“500 次”指 500 个训练 batch/参数更新，不是 500 张图片，也不是 epoch 内部计数。默认每个 epoch 有 938 次更新，20 个 epoch 共 18760 次。预览在 500、1000、…、18500 次触发；结束时额外生成 step 18760 的预览。随机抽样允许数字重复，不会把标签改成人工排列的 0–9。

常用覆盖参数：

```powershell
python -m DiTLab.train --epochs 5 --batch-size 64 --preview-every 500 --save-every 500
python -m DiTLab.train --output-dir DiTLab/outputs/run2 --cfg-scale 2.5
```

## 批量训练 20 组并对比曲线

在 `Step2` 目录运行：

```powershell
conda activate ml-practice
python -m DiTLab.batch --runs 20 --data-dir ../data --no-download
```

也可以在 `DiTLab` 中执行 `python batch.py --runs 20 --data-dir ../../data --no-download`。`--data-dir` 指包含 `MNIST` 子目录的路径；如使用默认 `DiTLab/data`，省略 `--data-dir` 即可。

这是 **20 组从头训练的独立实验**，每组沿用当前配置（目前为 20 个 epoch、学习率 `2e-5`），默认种子为 `42…61`。共训练 400 个 epoch。每组重新初始化模型和优化器，仅改变随机种子和输出目录。单卡上按顺序运行，每组使用当前激活环境的 Python 子进程；每组结束释放该进程的 GPU 资源。

首次运行时将所有参数保存到 `batch.json` 和 `configs/`，之后修改 `config.py` 不会改变这批剩余实验。预览和权重仍按每组内部的 500 次更新触发；每组只保留自己的最新 checkpoint。原有 `outputs/default` 不受影响。

```text
DiTLab/outputs/batch20/
  batch.json                 # 冻结的配置、种子和运行状态
  configs/                   # 各组完整配置
  run_001_seed42/
    train.log                # 完整训练日志，同时在终端输出
    losses.csv
    graph/loss_curve.png
    weights/latest.pt
    previews/
  run_002_seed43/
  ...
  comparison.png             # 各组 train/test 曲线及均值 ± 标准差
  combined_losses.csv        # 所有组的逐 epoch 原始 loss
  epoch_summary.csv          # 各 epoch 的有效组数 n、均值、样本标准差
  run_summary.csv            # 各组进度、最终/最小 loss、末值减首值
```

每组结束后刷新汇总。train 与 test 分成两个面板，同一颜色对应同一组；曲线不做平滑。未完成的组只贡献已有 epoch，不补零。标准差是样本标准差（单组时记为 0）；阴影不是置信区间。后面的 epoch 可能只有部分组完成，比较时同时看 `epoch_summary.csv` 中的 `n`。

中断或某组报错后，使用原计划继续，已完成组会自动跳过，未完成组从自己的最新 checkpoint 恢复：

```powershell
python -m DiTLab.batch --resume
```

恢复时不再重复 `--runs`、`--epochs`、`--data-dir` 等训练参数，保持原批次条件一致。无 checkpoint 的失败组会从头重启。失败时停止后续组，并将原因写入 `batch.json`；具体堆栈见对应组的 `train.log`。

训练中在另一个终端刷新已写入的 loss 对比，或训练结束后重新绘图：

```powershell
python -m DiTLab.batch --summary-only
```

另开一批或调整每组训练长度：

```powershell
python -m DiTLab.batch --runs 20 --epochs 5 --output-dir DiTLab/outputs/batch20_short --data-dir ../data --no-download
python -m DiTLab.batch --runs 20 --seed-start 100 --output-dir DiTLab/outputs/batch20_other --data-dir ../data --no-download
```

自定义输出目录的恢复/汇总也需加同一个 `--output-dir`。如只检查计划而暂不训练，加 `--plan-only`，之后用 `--resume` 启动。`--max-steps` 仅供短跑验证，不完成 epoch 时不会产生虚假的 loss 曲线。

Drift 使用归一化漂移场，loss 可能长期在接近 `9` 的范围波动，不保证单调下降；这些对比图反映相同目标在不同种子下的稳定性，不能只凭 loss 最低选出画质最好的模型，仍需结合 preview。

## 输出和加载

```text
DiTLab/outputs/default/
  config.json               # 本次实际配置
  losses.csv                # epoch, global_step, train_loss, test_loss
  graph/loss_curve.png       # 每个 epoch 更新的双折线图
  weights/latest.pt          # 唯一 checkpoint，原子覆盖
  previews/step_00000500.png # 同一张图中对齐 10 列
  previews/step_00000500.json# 测试集索引、原始标签、seed、采样配置
```

Preview 上排 `Test` 是测试图，下排 `Generated` 是对应数字条件下的生成结果。Drift 从独立高斯噪声一次生成，测试图只提供类别标签及显示参考；它不是逐图重建，所以同一列的笔迹无需相同。预览以 MNIST 的固定范围 `[-1,1]` 映射显示，训练时不裁剪模型输出。

checkpoint 包含模型、优化器、完整配置、epoch/batch/global step、当前 epoch 的 loss 累加值、历史 loss 和随机数状态。恢复时 `--epochs` 表示总目标 epoch 数：

```powershell
python -m DiTLab.train --resume
python -m DiTLab.train --resume DiTLab/outputs/default/weights/latest.pt --epochs 10
python -m DiTLab.sample
python -m DiTLab.sample --checkpoint DiTLab/outputs/run2/weights/latest.pt --cfg-scale 2.5
```

续训需保持原 CPU/CUDA 设备类型、batch size、seed 和学习率；新实验请使用新的输出目录。已有 checkpoint 的输出目录不会被一次新训练静默覆盖。独立采样可在 CPU 或 CUDA 加载同一权重，并按 checkpoint 配置自动重建模型，无需手填架构。

程序内只加载生成器，不加载数据集：

```python
import torch
from DiTLab.sample import load_generator

model, sampler, config, spec = load_generator("DiTLab/outputs/default/weights/latest.pt")
device = next(model.parameters()).device
labels = torch.arange(spec.num_classes, device=device)
noise = torch.randn(len(labels), spec.in_channels, spec.image_size, spec.image_size, device=device)
images = sampler.sample(model, noise, labels, cfg_scale=2.0)
```

## 算法说明

该实现参考本地 `Drift/drifting/drift_loss.py`、`train.py` 中的 Drifting，而非扩散噪声预测或 Flow Matching。模型接口为 `model(noise, labels, cfg_scale)`：DiT 用类别和 CFG 强度进行 adaLN 条件化，没有扩散时间步。采样严格只有一次模型前向；CFG 在训练中学入生成器，不做双前向输出外推。

每个真实 batch 按标签分组。同类真实图片作正样本；同类生成图片（排除自身）以及整个真实 batch 的无条件图片作负样本。后者权重为 `(cfg−1)×(G−1)/N`。由吸引/排斥场构建停止梯度的目标：`MSE(generated / scale, stopgrad(generated / scale + drift))`。距离尺度、双向 softmax 亲和度和各温度场归一化遵循参考实现。为适配轻量 MNIST 实验，使用展平像素特征及当前 batch，而没有 ImageNet 的特征网络或 memory bank；各类别分别归一化后按样本数加权。数值裁剪后再次精确清零自身亲和度，避免自身排斥。

**归一化 Drift loss 并非重建误差，也不保证随画质改善单调下降。** 每个温度的漂移场都做 RMS 归一化，三个温度合成的 loss 往往接近固定量级。折线图记录真实训练目标，应结合 preview 判断学习情况。这是 MNIST 像素空间适配实验，不宣称复现 ImageNet 论文指标。训练和测试使用相同目标；测试正负样本只来自测试集，不反向传播，并固定评估随机数使各 epoch 可比。预览和评估不会消耗训练随机数流。

## 替换组件

- `models/dit.py`：纯模型，仅接收噪声、类别条件与指导强度并返回图像。
- `algorithms/drift.py`：训练目标和一步采样实现；训练算法、采样器在配置中独立选择。
- `data.py`：返回 `DataBundle(train_dataset, test_dataset, DataSpec)`；每条数据为 `(CHW tensor, integer label)`。
- `interfaces.py`：训练 `loss(...)` 和采样 `sample(...)` 的最小协议。
- `registry.py`：四个 builder 注册表，是新增/替换组件的统一接线位置。
- `engine.py`：训练/测试、计数、preview 调度及断点恢复，不引用 DiT 或 Drift。
- `artifacts.py`、`preview.py`：checkpoint、loss CSV/曲线和预览渲染。
- `config.py`：配置及默认值；`train.py`、`sample.py`：命令行入口。

新增组件时实现相应协议、在注册表添加名称，然后使用 JSON 切换，例如：

```json
{
  "model": {"name": "dit", "options": {"hidden_size": 256, "num_heads": 8, "depth": 6}},
  "algorithm": {"name": "drift", "options": {"generated_per_class": 16}},
  "sampler": {"name": "drift", "options": {"cfg_scale": 2.5}},
  "training": {"epochs": 5, "output_dir": "DiTLab/outputs/larger"}
}
```

保存为配置文件后运行 `python -m DiTLab.train --config 路径.json`。同名组件未写出的选项沿用默认值，更换组件名称时不继承旧组件参数。换数据集时也需让模型与训练算法支持新的图像/标签接口。当前训练和测试均不含随机数据增强；若添加随机增强，需相应扩展恢复逻辑以保存其随机状态。

## 验证

```powershell
conda activate ml-practice
python -m unittest discover -s DiTLab/tests -v
```

测试使用小型合成数据，不触发下载。另可用 `--max-steps 2 --output-dir DiTLab/outputs/smoke` 做真实数据短跑；若未完成完整 epoch，不会伪造 epoch loss。正式实验不要设置 `--max-steps`。

可复现的真实 MNIST 小样本集成验证（128 张训练图、40 张测试图，2 个短 epoch，保持默认 DiT 架构）：

```powershell
python -m DiTLab.tests.smoke_mnist --data-dir ../data
```

当前已在 `ml-practice`（PyTorch 2.11.0+cu128、RTX 5070 Laptop GPU）通过该短跑，检查了 loss CSV/折线图、10 图预览、唯一 checkpoint 以及加载后生成；另外验证了 CUDA 权重在 CPU 上独立采样。42 项单元/集成测试全部通过，包括中途恢复与连续训练的权重和随机状态一致性、批次跳过/恢复、防误覆盖、汇总均值和标准差。批量入口也通过两组 CUDA 短跑及恢复验证（每组先运行 1 次更新，再恢复到 2 次更新）。`outputs/smoke_mnist` 和 `outputs/batch_smoke` 都是验证结果；没有将它们作为正式的完整训练结果。

Drift 改编部分的 MIT 版权声明保留于 `algorithms/LICENSE`。
