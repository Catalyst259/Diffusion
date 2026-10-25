"""Render held-out references and label-conditioned generations as one PNG."""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import torch


def _display_images(
    batch: torch.Tensor, value_min: float, value_max: float
) -> list[Image.Image]:
    pixels = batch.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(pixels).all():
        raise ValueError("Preview images must contain only finite values.")
    pixels = ((pixels - value_min) / (value_max - value_min)).clamp(0, 1)
    pixels = (pixels * 255).round().to(torch.uint8)
    result = []
    for tensor in pixels:
        if tensor.shape[0] == 1:
            image = Image.fromarray(tensor[0].numpy()).convert("RGB")
        else:
            image = Image.fromarray(tensor.permute(1, 2, 0).numpy())
        result.append(image)
    return result


def save_preview(
    reference: torch.Tensor,
    generated: torch.Tensor,
    labels: torch.Tensor,
    path: str | Path,
    *,
    value_min: float = -1.0,
    value_max: float = 1.0,
    title: str = "",
) -> Path:
    """Save two aligned rows with native digit labels above each column.

    Reference images are for comparison only: the sampler receives their
    labels and independent noise, not the reference image pixels.
    """
    if reference.ndim != 4 or reference.shape != generated.shape:
        raise ValueError("Reference and generated images must have equal NCHW shapes.")
    count, channels, height, width = reference.shape
    if count < 1 or height < 1 or width < 1 or channels not in (1, 3):
        raise ValueError("Preview expects a nonempty grayscale or RGB batch.")
    if labels.ndim != 1 or len(labels) != count:
        raise ValueError("Provide one class label per preview column.")
    if not (math.isfinite(value_min) and math.isfinite(value_max)) or value_max <= value_min:
        raise ValueError("Image value_max must be finite and greater than value_min.")

    references = _display_images(reference, value_min, value_max)
    generations = _display_images(generated, value_min, value_max)
    class_labels = labels.detach().cpu().tolist()
    scale = max(1, 112 // max(height, width))
    tile_width, tile_height = width * scale, height * scale
    column_width = max(tile_width, 80)
    margin, gap, row_label_width, header_height = 12, 10, 88, 24
    title_height = 24 if title else 0
    image_left = margin + row_label_width
    image_top = margin + title_height + header_height
    panel = Image.new(
        "RGB",
        (
            image_left + count * column_width + (count - 1) * gap + margin,
            image_top + 2 * tile_height + gap + margin,
        ),
        color=(235, 235, 235),
    )
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:  # Pillow before scalable default fonts were introduced.
        font = ImageFont.load_default()
    if title:
        draw.text((margin, margin), title, fill="black", font=font)
    for row, (row_name, images) in enumerate(
        (("Test", references), ("Generated", generations))
    ):
        top = image_top + row * (tile_height + gap)
        draw.text((margin, top + tile_height // 2 - 7), row_name, fill="black", font=font)
        for column, image in enumerate(images):
            left = image_left + column * (column_width + gap)
            panel.paste(
                image.resize((tile_width, tile_height), resample=Image.Resampling.NEAREST),
                (left + (column_width - tile_width) // 2, top),
            )
            if row == 0:
                draw.text(
                    (left, margin + title_height),
                    f"Label {int(class_labels[column])}",
                    fill="black",
                    font=font,
                )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    panel.save(destination, format="PNG")
    return destination
