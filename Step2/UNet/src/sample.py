from pathlib import Path

import math
from numbers import Integral, Real

import torch
from PIL import Image, ImageDraw


# t=0 是干净图；扩散时间为 1..1000，系数表的第 0 项不加噪。
NUM_TIMESTEPS = 1000
betas = torch.cat((torch.zeros(1), torch.linspace(1e-4, 0.02, NUM_TIMESTEPS)))
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

betas_ = betas.view(-1, 1, 1, 1)
alphas_ = alphas.view(-1, 1, 1, 1)
alphas_cumprod_ = alphas_cumprod.view(-1, 1, 1, 1)
sqrt_alphas_cumprod_ = alphas_cumprod_.sqrt()
sqrt_one_minus_alphas_cumprod_ = (1 - alphas_cumprod_).sqrt()
_DEVICE_SCHEDULES = {}


def _extract(values, t):
    if values.device == t.device:
        return values[t]
    key = (id(values), t.device)
    if key not in _DEVICE_SCHEDULES:
        _DEVICE_SCHEDULES[key] = values.to(t.device)
    return _DEVICE_SCHEDULES[key][t]

# 生成 xt
def q_sample(x0, t, noise=None):
    """前向加噪：x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * noise。"""
    if noise is None:
        noise = torch.randn_like(x0)
    return (
        _extract(sqrt_alphas_cumprod_, t) * x0
        + _extract(sqrt_one_minus_alphas_cumprod_, t) * noise
    )

# 计算后验分布均值 (miu_tilde)
def q_posterior_mean(x0, xt, t):
    """q(x_{t-1} | x_t, x0) 的均值，要求 1 <= t <= 1000。"""
    a = _extract(alphas_, t)
    a_bar = _extract(alphas_cumprod_, t)
    a_bar_prev = _extract(alphas_cumprod_, t - 1)
    b = _extract(betas_, t)
    mean = (
        a_bar_prev.sqrt() * b * x0
        + a.sqrt() * (1 - a_bar_prev) * xt
    ) / (1 - a_bar)
    # 避免浮点误差；最后一步直接得到预测的 x0。
    return torch.where((t == 1).view(-1, 1, 1, 1), x0, mean)

# 计算后验分布方差 (sigma_tilde^2)
def q_posterior_variance(t):
    """q(x_{t-1} | x_t, x0) 的方差；t=1 时为 0。"""
    a_bar = _extract(alphas_cumprod_, t)
    a_bar_prev = _extract(alphas_cumprod_, t - 1)
    b = _extract(betas_, t)
    return b * (1 - a_bar_prev) / (1 - a_bar)

# 生成标签
def eps_sample(x0, xt, t):
    """还原前向过程加入的累计噪声，作为 epsilon 模型的训练目标。"""
    return (
        xt - _extract(sqrt_alphas_cumprod_, t) * x0
    ) / _extract(sqrt_one_minus_alphas_cumprod_, t)

def _batch_indices(values, images, name):
    values = torch.as_tensor(values, device=images.device)
    if values.is_complex() or not torch.isfinite(values).all():
        raise ValueError(f"{name} must contain finite integers")
    if values.is_floating_point() and (values != values.round()).any():
        raise ValueError(f"{name} must contain integers")
    values = values.to(dtype=torch.long)
    if values.ndim == 0:
        values = values.expand(images.size(0))
    if values.shape != (images.size(0),) or values.numel() == 0:
        raise ValueError(f"{name} must contain one value per image")
    return values


def _ddim_grid(t, num_steps):
    """Each column is one trajectory, padded with clean t=0 after its last step."""
    grids = [
        torch.linspace(start, 0, min(start, num_steps) + 1, dtype=torch.float64)
        .round().long().unique_consecutive()
        for start in t.detach().cpu().tolist()
    ]
    grid = torch.zeros(max(len(column) for column in grids), len(grids), dtype=torch.long)
    for index, column in enumerate(grids):
        grid[:len(column), index] = column
    return grid.to(t.device)


def _random_noise(images, generator):
    # Preserve the original RNG call when no explicit generator is supplied.
    if generator is None:
        return torch.randn_like(images)
    return torch.randn(images.shape, dtype=images.dtype, device=images.device, generator=generator)


@torch.no_grad()
def inference(xt, t, model, *, on_step=None, c=None, sampler="ddpm", num_steps=50,
              eta=0.0, generator=None, cfg_scale=1.0):
    """Sample to clean t=0 with DDPM or DDIM, including mixed initial timesteps.

    DDPM retains the original full trajectory. DDIM uses at most num_steps
    denoising updates; CFG uses two predictions per guided update.
    eta=0 is deterministic and eta=1 uses posterior
    noise. c optionally supplies class labels to a three-input model.
    cfg_scale=1 preserves the single conditional prediction; other scales
    combine conditional and explicit null-label predictions from a model
    trained with label dropout (0 is unconditional, >1 strengthens guidance).
    on_step(images, timesteps) receives the initial state and every update.
    Inputs and the model's original train/eval mode are preserved.
    """
    if not isinstance(sampler, str) or sampler.lower() not in ("ddpm", "ddim"):
        raise ValueError("sampler must be 'ddpm' or 'ddim'")
    sampler = sampler.lower()
    if isinstance(num_steps, bool) or not isinstance(num_steps, Integral) or not 1 <= num_steps <= NUM_TIMESTEPS:
        raise ValueError(f"num_steps must be an integer between 1 and {NUM_TIMESTEPS}")
    if isinstance(eta, bool) or not isinstance(eta, Real) or not math.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError("eta must be a finite number between 0 and 1")
    if (isinstance(cfg_scale, bool) or not isinstance(cfg_scale, Real)
            or not math.isfinite(cfg_scale) or cfg_scale < 0):
        raise ValueError("cfg_scale must be a finite nonnegative number")
    t = _batch_indices(t, xt, "t")
    if ((t < 0) | (t > NUM_TIMESTEPS)).any():
        raise ValueError(f"t must be between 0 and {NUM_TIMESTEPS}")
    labels = None if c is None else _batch_indices(c, xt, "c")
    null_label = None
    if cfg_scale != 1:
        if labels is None:
            raise ValueError("CFG requires explicit class labels c")
        num_classes = getattr(model, "num_classes", None)
        null_label = getattr(model, "null_label", None)
        if (isinstance(num_classes, bool) or not isinstance(num_classes, Integral)
                or num_classes <= 0 or isinstance(null_label, bool)
                or not isinstance(null_label, Integral) or null_label < num_classes):
            raise ValueError("CFG requires a model trained with label dropout and a null-label "
                             "embedding (num_classes and null_label); this model does not support CFG")
        if ((labels < 0) | (labels >= num_classes)).any():
            raise ValueError(f"CFG class labels c must be between 0 and {num_classes - 1}")
    grid = _ddim_grid(t, num_steps) if sampler == "ddim" else None
    steps = len(grid) - 1 if grid is not None else int(t.max().item())

    images = xt.clone()
    was_training = model.training
    model.eval()
    try:
        if on_step is not None:
            on_step(images, t)
        for index in range(steps):
            active = t > 0
            active_t = t[active]
            active_images = images[active]
            predicted_eps = (model(active_images, active_t) if labels is None
                             else model(active_images, active_t, labels[active]))
            if cfg_scale != 1:
                # Separate forwards avoid doubling the peak activation batch size.
                # Every image channel is epsilon (MNIST has one); guide all of them.
                null_labels = torch.full_like(active_t, null_label)
                unconditional_eps = model(active_images, active_t, null_labels)
                predicted_eps = unconditional_eps + cfg_scale * (predicted_eps - unconditional_eps)
            predicted_x0 = (
                active_images
                - _extract(sqrt_one_minus_alphas_cumprod_, active_t) * predicted_eps
            ) / _extract(sqrt_alphas_cumprod_, active_t)
            next_t = (t - 1).clamp_min(0) if grid is None else grid[index + 1]
            if sampler == "ddpm":
                mean = q_posterior_mean(predicted_x0, active_images, active_t)
                variance = q_posterior_variance(active_t)
                images[active] = mean + variance.sqrt() * _random_noise(active_images, generator)
            else:
                active_next_t = next_t[active]
                a_bar = _extract(alphas_cumprod_, active_t)
                a_bar_next = _extract(alphas_cumprod_, active_next_t)
                variance = eta ** 2 * (1 - a_bar_next) / (1 - a_bar) * (1 - a_bar / a_bar_next)
                variance = variance.clamp_min(0)
                mean = (a_bar_next.sqrt() * predicted_x0
                        + (1 - a_bar_next - variance).clamp_min(0).sqrt() * predicted_eps)
                if eta == 1:
                    # Exact original arithmetic for consecutive DDPM-equivalent steps.
                    adjacent = (active_next_t == active_t - 1).view(-1, 1, 1, 1)
                    mean = torch.where(adjacent, q_posterior_mean(predicted_x0, active_images, active_t), mean)
                    variance = torch.where(adjacent, q_posterior_variance(active_t), variance)
                images[active] = mean if eta == 0 else mean + variance.sqrt() * _random_noise(active_images, generator)
            t = next_t
            if on_step is not None:
                on_step(images, t)
    finally:
        model.train(was_training)
    return images


def save_inference_gif(xt, t, model, path, *, frame_every=10, duration_ms=80,
                       c=None, sampler="ddpm", num_steps=50, eta=0.0, generator=None,
                       cfg_scale=1.0):
    """保存单张 MNIST 图从 t 到 0 的 GIF，并返回最终 x0。

    所有去噪步骤都会执行；每 frame_every 步保存一帧，始终保留首尾帧。
    显示范围固定为 [0, 1]，与 ToTensor() 一致，不对每帧单独归一化。
    """
    if xt.ndim != 4 or xt.shape[:2] != (1, 1):
        raise ValueError("GIF preview expects one grayscale image: [1, 1, H, W]")
    if frame_every < 1 or duration_ms < 1:
        raise ValueError("frame_every and duration_ms must be positive")
    frames = []
    step_index = -1

    def capture(images, timesteps):
        nonlocal step_index
        step_index += 1
        current_t = int(timesteps[0].item())
        if step_index != 0 and current_t != 0 and step_index % frame_every:
            return
        pixels = images[0, 0].detach().clamp(0, 1).mul(255).round().byte().cpu().numpy()
        picture = Image.fromarray(pixels)
        picture = picture.resize((picture.width * 8, picture.height * 8), Image.Resampling.NEAREST)
        frame = Image.new("RGB", (max(224, picture.width), picture.height + 32), "white")
        frame.paste(picture, ((frame.width - picture.width) // 2, 32))
        label = f"t = {current_t}" + ("   (x0)" if current_t == 0 else "")
        ImageDraw.Draw(frame).text((8, 8), label, fill="black")
        frames.append(frame)

    x0 = inference(xt, t, model, on_step=capture, c=c, sampler=sampler,
                   num_steps=num_steps, eta=eta, generator=generator, cfg_scale=cfg_scale)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    durations = [duration_ms] * len(frames)
    durations[-1] = max(800, duration_ms)
    frames[0].save(
        path, format="GIF", save_all=True, append_images=frames[1:],
        duration=durations, loop=0, disposal=2,
    )
    return x0
