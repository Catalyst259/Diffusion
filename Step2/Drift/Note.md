Drifting Model 意图训练一个直接的映射，在像素维度层面从标准正态分布映射到真实图片分布，训练的核心思路是让原本正态分布经过 f 后变换得到的新分布尽可能贴近图像的原始分布

Traditional generative models (Diffusion, Flow Matching) iterate at inference time.

Drifting Models move the iteration to training time:

- Each training step "drifts" the generated samples according to a drifting field
- At inference: just one forward pass (1 NFE)

`L=Eϵ[||fθ(ϵ)−stopgrad(fθ(ϵ)+Vp,q(fθ(ϵ)))||^2]`
第一个 fθ(ϵ) 是模型根据输入噪声 ϵ 生成的样本，第二个 fθ(ϵ)+Vp,q(fθ(ϵ)) 是在生成样本的基础上应用漂移场 Vp,q 的结果。如果不加 stopgrad，模型会让 V^2 尽可能小，即优化模长，从而导致模型退化为一个恒等映射，无法学习到有效的漂移场。
加上 stopgrad，让模型沿向量场漂移
现在有了训练目标，我们要构造 V 了
