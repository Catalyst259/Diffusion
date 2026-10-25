第二阶段：带 cfg 进阶实验，DiT 模型架构升级，与 Drifting 性能评估
前置条件：学习 DiT 模型架构与 Drifting 算法。
1. 快速梯度爆炸，随后的验证集显示模型未学到任何模式，反而会把未加噪完全的图像破坏
解决方案：学习率 1e-3 -> 1e-4
2. UNet 的时间条件直接加 cfg 信息效果非常差
3. 使用 DiT 模型 + Drifting 算法，在像素空间内清晰度表现较差，但是对平均情况拟合得非常好，反观去噪算法 + UNet 虽然有较好的清晰度，但是偏差依然较大
4. DiT + EDM 第一轮训练失败，图像碎片化。加大高步数去噪比例后问题缓解
![](DiTLab/outputs/dit57m_compare/drift_vs_edm.png)
Drifting 的生成结果明显更加稳定，但是清晰度不高；EDM 的生成结果清晰度较高，但是在平均情况拟合上不如 Drifting。
对比实验使用的参数量和 UNet 相同，Drifting 的训练时间是 EDM 的 1.5 倍。（Drifting 16 分钟，EDM 10 分钟）
由于 Drifting 只需 1 次前向，EDM 共用了 63 次前向，单卡推理 Drifting 是 EDM 的 60 倍，整批耗时 EDM 约为 Drifting 的 80 倍（DiTLab/outputs/dit57m_compare/inference_benchmark.json）