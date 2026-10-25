import math

import torch
from torch import nn
from torch.nn import functional as F
# 手搓的
class TimeEmbedding(nn.Module):
    """
    Time embedding layer for the UNet model.
    Attributes:
        time_emb_dim (int): The dimension of the time embedding.
    """
    def __init__(self, time_emb_dim: int):
        super().__init__()
        self.time_emb_dim = time_emb_dim

    def forward(self, t): # [B]
        """
        Forward pass of the time embedding layer.
        Args:
            t (torch.Tensor): The input time tensor. Shape: [B]
        Returns:
            torch.Tensor: The time embedding. Shape: [B, time_emb_dim]
        """
        half_dim = self.time_emb_dim // 2
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * (-math.log(10000.0) / (half_dim - 1))) # [half_dim]  e^(2i/d * -log(10000))
        emb = t[:, None] * emb[None, :] # [B, half_dim]  e^(2i/d * -log(10000)) * t
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) # [B, time_emb_dim]
        return emb  # [B, time_emb_dim]

class ResBlock(nn.Module):
    """
    Residual block for the UNet model.
    Keeps the same height and width of the input tensor.
    For adding time and Residual information
    Attributes:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        time_emb_dim (int): The dimension of the time embedding.
        mid_channels (int): The number of middle channels.
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, mid_channels = None):
        super().__init__()
        self.act = nn.SiLU()
        mid_channels = mid_channels or out_channels
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(nn.SiLU(),nn.Linear(time_emb_dim, mid_channels),)

        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        """
        Forward pass of the residual block.
        Args:
            x (torch.Tensor): The input tensor. Shape: [B, in_channels, H, W]
            t_emb (torch.Tensor): The time embedding tensor. Shape: [B, time_emb_dim]
        Returns:
            torch.Tensor: The output tensor. Shape: [B, out_channels, H, W]
        """

        h = self.act(self.norm1(x)) # [B, in_channels, H, W]
        h = self.conv1(h) # [B, mid_channels, H, W]
        t = self.time_mlp(t_emb)[:, :, None, None]  # [B, mid_channels, 1, 1]
        h = h + t # [B, mid_channels, H, W]
        h = self.act(self.norm2(h)) # [B, mid_channels, H, W]
        h = self.conv2(h) # [B, out_channels, H, W]
        return h + self.skip(x) # [B, out_channels, H, W]

class DownSample(nn.Module):
    """
    Downsampling block for the UNet model.
    In DDPM, channel doesnot change after downsampling, so in_channels = out_channels
    Attributes:
        channels (int): The number of channels.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """
        Forward pass of the downsampling block.
        Args:
            x (torch.Tensor): The input tensor. Shape: [B, channels, H, W]
        Returns:
            torch.Tensor: The output tensor. Shape: [B, channels, H/2, W/2]
        """
        return self.conv(x) # [B, channels, H/2, W/2]

class UpSample(nn.Module):
    """
    Upsampling block for the UNet model.
    In DDPM, channel doesnot change after upsampling, so in_channels = out_channels
    Attributes:
        channels (int): The number of channels.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, output_size=None):
        """
        Forward pass of the upsampling block.
        Args:
            x (torch.Tensor): The input tensor. Shape: [B, channels, H, W]
            output_size (tuple, optional): Spatial size of the matching skip connection.
        Returns:
            torch.Tensor: The output tensor at output_size or [B, channels, H*2, W*2].
        """
        x = self.upsample(x) if output_size is None else F.interpolate(x, size=output_size, mode='nearest')
        return self.conv(x) # [B, channels, H*2, W*2]

class LabelEmbedder(nn.Embedding):
    """DiT label dropout, keeping the legacy ``label_embed.weight`` key.

    A positive dropout probability adds one learned null-class embedding. Each
    training example independently uses it with that probability. Evaluation
    keeps the supplied labels unless an explicit drop mask is provided.
    """

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__(num_classes + int(dropout_prob > 0), hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.null_label = num_classes if dropout_prob > 0 else None

    def forward(self, labels, *, force_drop_ids=None):
        if force_drop_ids is not None:
            if not isinstance(force_drop_ids, torch.Tensor) or force_drop_ids.shape != labels.shape:
                raise ValueError("force_drop_ids must be a tensor of shape [B]")
            if force_drop_ids.device != labels.device:
                raise ValueError("force_drop_ids and c must be on the same device")
            if torch.any((force_drop_ids != 0) & (force_drop_ids != 1)):
                raise ValueError("force_drop_ids must contain only 0 or 1")
            drop_ids = force_drop_ids == 1
            if self.null_label is None and torch.any(drop_ids):
                raise ValueError("label dropout requires a learned null embedding (class_dropout_prob > 0)")
        elif self.training and self.dropout_prob > 0:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            # In particular, p=0 preserves the legacy random-number stream.
            return super().forward(labels)
        labels = torch.where(drop_ids, self.num_classes, labels)
        return super().forward(labels)


class UNet(nn.Module):
    """
    UNet model for the DDPM.
    Attributes:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        time_emb_dim (int): The dimension of the time embedding.
        base_channels (int): The number of base channels.
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, base_channels: int = 64, num_classes: int | None = None, class_dropout_prob: float = 0.1):
        super().__init__()
        if num_classes is not None and (isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 0):
            raise ValueError("num_classes must be a positive integer or None")
        if isinstance(class_dropout_prob, bool) or not isinstance(class_dropout_prob, (int, float)) or not 0 <= class_dropout_prob <= 1:
            raise ValueError("class_dropout_prob must be a number between 0 and 1")
        self.num_classes = num_classes
        self.class_dropout_prob = float(class_dropout_prob)
        self.null_label = num_classes if num_classes is not None and class_dropout_prob > 0 else None
        self.time_mlp = nn.Sequential(TimeEmbedding(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim * 4), nn.SiLU(), nn.Linear(time_emb_dim * 4, time_emb_dim))
        # DiT-style conditioning: add the learned class embedding to the time
        # encoding, then reuse the existing injection in every residual block.
        # None registers no extra parameters, preserving legacy state_dict keys.
        # p=0 retains the ten-row Turn3 format; positive p reserves the extra
        # null row required for classifier-free guidance. Checkpoint loading
        # stays strict so old weights cannot silently acquire an untrained row.
        self.label_embed = None if num_classes is None else LabelEmbedder(num_classes, time_emb_dim, self.class_dropout_prob)
        if self.label_embed is not None:
            nn.init.normal_(self.label_embed.weight, std=0.02)

        self.conv0 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.encoder1 = ResBlock(base_channels, base_channels * 2, time_emb_dim)
        self.encoder2 = ResBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.encoder3 = ResBlock(base_channels * 4, base_channels * 8, time_emb_dim)
        self.downsample1 = DownSample(base_channels * 2)
        self.downsample2 = DownSample(base_channels * 4)
        self.downsample3 = DownSample(base_channels * 8)

        self.mid1 = ResBlock(base_channels * 8, base_channels * 8, time_emb_dim)
        self.mid2 = ResBlock(base_channels * 8, base_channels * 8, time_emb_dim)

        self.upsample1 = UpSample(base_channels * 8)
        self.decoder1 = ResBlock(base_channels * 16, base_channels * 4, time_emb_dim)
        self.upsample2 = UpSample(base_channels * 4)
        self.decoder2 = ResBlock(base_channels * 8, base_channels * 2, time_emb_dim)
        self.upsample3 = UpSample(base_channels * 2)
        self.decoder3 = ResBlock(base_channels * 4, base_channels, time_emb_dim)

        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, t, c=None, *, force_drop_ids=None):
        """
        Forward pass of the UNet model.
        Args:
            x (torch.Tensor): The input tensor. Shape: [B, in_channels, H, W]
            t (torch.Tensor): The time tensor. Shape: [B]
            c (torch.Tensor, optional): Integer class labels. Shape: [B].
                Required when constructed with num_classes; omit for legacy models.
                Use null_label explicitly for unconditional CFG predictions.
            force_drop_ids (torch.Tensor, optional): Per-example 0/1 mask,
                overriding random label dropout in both train and eval modes.
        Returns:
            torch.Tensor: The output tensor. Shape: [B, out_channels, H, W]
        """
        t_emb = self.time_mlp(t) # [B, time_emb_dim]
        if self.label_embed is not None:
            if c is None:
                raise ValueError("c is required for a class-conditional UNet")
            if c.shape != (x.shape[0],) or c.dtype != torch.long:
                raise ValueError("c must be a torch.long tensor of shape [B]")
            if c.device != x.device:
                raise ValueError("c and x must be on the same device")
            t_emb = t_emb + self.label_embed(c, force_drop_ids=force_drop_ids)
        elif c is not None:
            raise ValueError("c requires a UNet constructed with num_classes")
        elif force_drop_ids is not None:
            raise ValueError("force_drop_ids requires a class-conditional UNet")

        x0 = self.conv0(x) # [B, base_channels, H, W]

        x1 = self.encoder1(x0, t_emb) # [B, base_channels*2, H, W]
        x1_down = self.downsample1(x1) # [B, base_channels*2, H/2, W/2]

        x2 = self.encoder2(x1_down, t_emb) # [B, base_channels*4, H/2, W/2]
        x2_down = self.downsample2(x2) # [B, base_channels*4, H/4, W/4]

        x3 = self.encoder3(x2_down, t_emb) # [B, base_channels*8, H/4, W/4]
        x3_down = self.downsample3(x3) # [B, base_channels*8, H/8, W/8]

        x_mid = self.mid1(x3_down, t_emb) # [B, base_channels*16, H/8, W/8]
        x_mid = self.mid2(x_mid, t_emb) # [B, base_channels*8, H/8, W/8]

        x_up1 = self.upsample1(x_mid, x3.shape[-2:]) # [B, base_channels*8, H/4, W/4]
        x_up1 = torch.cat((x_up1, x3), dim=1) # [B, base_channels*16, H/4, W/4]
        x_up1 = self.decoder1(x_up1, t_emb) # [B, base_channels*4, H/4, W/4]

        x_up2 = self.upsample2(x_up1, x2.shape[-2:]) # [B, base_channels*4, H/2, W/2]
        x_up2 = torch.cat((x_up2, x2), dim=1) # [B, base_channels*8, H/2, W/2]
        x_up2 = self.decoder2(x_up2, t_emb) # [B, base_channels*2, H/2, W/2]

        x_up3 = self.upsample3(x_up2, x1.shape[-2:]) # [B, base_channels*2, H, W]
        x_up3 = torch.cat((x_up3, x1), dim=1) # [B, base_channels*4, H, W]
        x_up3 = self.decoder3(x_up3, t_emb) # [B, base_channels, H, W]

        x_out = self.conv_out(x_up3) # [B, out_channels, H, W]
        return x_out # [B, out_channels, H, W]


if __name__ == "__main__":
    m = ResBlock(in_channels=64, out_channels=128, time_emb_dim=256)
    x = torch.randn(2, 64, 32, 32)
    t = torch.randn(2, 256)
    y = m(x, t)
    print(y.shape)   # torch.Size([2, 128, 32, 32])
