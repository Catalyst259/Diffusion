from pathlib import Path
import math
from math import sqrt
import torch
from torch import nn
from PIL import Image


def preprocess(images_dir: str | Path = "images", batch_size: int = 8) -> torch.Tensor:
    """Load RGB images into a `(B, 3, 1024, 768)` float tensor.

    Images are resized to height 1024 and width 768.  When the directory has
    fewer images than ``batch_size``, its images are repeated in sorted order
    so that the returned batch always has the requested size.
    """
    images_path = Path(images_dir)
    image_paths = sorted(
        path for path in images_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {images_path}")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    batch = []
    for index in range(batch_size):
        with Image.open(image_paths[index % len(image_paths)]) as image:
            # PIL expects (width, height); convert discards any alpha channel.
            image = image.convert("RGB").resize((768, 1024), Image.Resampling.LANCZOS)
            tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            tensor = tensor.reshape(1024, 768, 3).permute(2, 0, 1)
            batch.append(tensor)

    return torch.stack(batch).to(torch.float32).div_(255.0)


class ConvBlock(nn.Module):
    """Two convolution layers with GroupNorm, which is stable for small batches."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        groups = 4 if out_channels % 4 == 0 else 1
        self.conv1: nn.Conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1: nn.GroupNorm = nn.GroupNorm(groups, out_channels)
        self.act1: nn.SiLU = nn.SiLU()
        self.time_proj: nn.Linear = nn.Linear(time_dim, out_channels)
        self.conv2: nn.Conv2d = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2: nn.GroupNorm = nn.GroupNorm(groups, out_channels)
        self.act2: nn.SiLU = nn.SiLU()



    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """单纯做加法把时间加上去"""
        h = self.act1(self.norm1(self.conv1(x)))
        time_features = self.time_proj(t)[:, :, None, None]
        h = h + time_features
        h = self.act2(self.norm2(self.conv2(h)))
        return h

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion models."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10_000) / (half_dim - 1)

        frequencies = torch.exp(
            -scale * torch.arange(
                half_dim, device=t.device, dtype=torch.float32
            )
        )

        angles = t.float()[:, None] * frequencies[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=1)

class UNet(nn.Module):
    """A compact U-Net for `(B, 3, 1024, 768)` image tensors.

    With the default ``base_channels=8``, this model has roughly 122K
    trainable parameters. Its output is unconstrained, which suits predicting
    noise/residuals in a diffusion model; apply an activation only if the task
    instead requires RGB pixels in a fixed range.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 8) -> None:
        super().__init__()
        c1, c2, c3, c4 = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        time_dim = 32  # You can adjust this as needed
        # 完整 Embedding，做一次变换 + 投射
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.encoder1 = ConvBlock(in_channels, c1, time_dim)
        self.encoder2 = ConvBlock(c1, c2, time_dim)
        self.encoder3 = ConvBlock(c2, c3, time_dim)
        self.bottleneck = ConvBlock(c3, c4, time_dim)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.in_channels = in_channels
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(c3 * 2, c3, time_dim)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(c2 * 2, c2, time_dim)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(c1 * 2, c1, time_dim)
        self.output = nn.Conv2d(c1, out_channels, kernel_size=1)

    # 输出形状为 x 的噪声
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        assert t.ndim == 1 and t.shape[0] == x.shape[0], "t must be a 1D tensor with the same batch size as x"
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("Expected an input tensor shaped (B, 3, H, W)")
        if x.shape[-2] % 8 != 0 or x.shape[-1] % 8 != 0:
            raise ValueError("Height and width must be divisible by 8")
        t_emb = self.time_embedding(t)
        skip1 = self.encoder1(x, t_emb)
        skip2 = self.encoder2(self.pool(skip1), t_emb)
        skip3 = self.encoder3(self.pool(skip2), t_emb)
        encoded = self.bottleneck(self.pool(skip3), t_emb)

        decoded = self.decoder3(torch.cat((self.up3(encoded), skip3), dim=1), t_emb)
        decoded = self.decoder2(torch.cat((self.up2(decoded), skip2), dim=1), t_emb)
        decoded = self.decoder1(torch.cat((self.up1(decoded), skip1), dim=1), t_emb)
        return self.output(decoded)

def q_sample(x0, t, alpha_bars, noise=None):
    """针对 t 时刻进行采样，得到 t 时刻的噪声和图像"""
    if noise is None:
        # 如果没有提供噪声，就生成一个与 x0 形状相同的标准正态分布噪声
        noise = torch.randn_like(x0)
    alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)

    xt = (
        alpha_bar_t.sqrt() * x0
        + (1 - alpha_bar_t).sqrt() * noise
    )
    return xt, noise


def train_on_batch(
    model: nn.Module,
    images: torch.Tensor,
    alpha_bars: torch.Tensor,
    steps: int = 5,
    learning_rate: float = 1e-3,
) -> list[float]:
    """Overfit ``model`` to one image batch for a small number of steps."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    model.train()
    for _ in range(steps):
        t = torch.randint(0, len(alpha_bars), (images.shape[0],), dtype=torch.long)
        xt, noise = q_sample(images, t, alpha_bars)
        predicted_noise = model(xt, t)
        loss = nn.functional.mse_loss(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def predicted_x0(
    xt: torch.Tensor, predicted_noise: torch.Tensor, t: torch.Tensor, alpha_bars: torch.Tensor
) -> torch.Tensor:
    """Convert a predicted diffusion noise tensor back to an RGB image estimate."""
    alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
    return (xt - (1 - alpha_bar_t).sqrt() * predicted_noise) / alpha_bar_t.sqrt()


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Decode one `(3, H, W)` tensor in `[0, 1]` into a PIL RGB image."""
    tensor = tensor.detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8)
    height, width = tensor.shape[-2:]
    pixels = tensor.permute(1, 2, 0).contiguous().numpy().tobytes()
    return Image.frombytes("RGB", (width, height), pixels)


if __name__ == "__main__":
    # Example usage
    T = 100
    # 把图片变成 (B, 3, 768, 1024) 的张量
    images = preprocess("images", batch_size=8) # (B, 3, 768, 1024)
    # 计算 beta, alpha, alpha_bar
    betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    model = UNet()
    losses = train_on_batch(model, images, alpha_bars, steps=10)
    print(f"loss after 10 steps: {losses[-1]:.6f}")

    image_index = torch.randint(0, images.shape[0], ()).item()
    x0 = images[image_index : image_index + 1]
    t = torch.randint(0, T, (1,), dtype=torch.long)
    xt, _ = q_sample(x0, t, alpha_bars)
    model.eval()
    with torch.no_grad():
        output = predicted_x0(xt, model(xt, t), t, alpha_bars)

    output_image = tensor_to_image(output[0])
    output_path = Path("denoised_output.png")
    output_image.save(output_path)
    print(f"saved decoded output to: {output_path.resolve()}")
    output_image.show()
