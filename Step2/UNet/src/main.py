from pathlib import Path

import csv
import json

import torch
from torch import nn
if __package__:
    from .model import UNet
    from .sample import NUM_TIMESTEPS, q_sample, save_inference_gif
else:
    from model import UNet
    from sample import NUM_TIMESTEPS, q_sample, save_inference_gif


PREVIEW_EVERY = 100
GIF_FRAME_EVERY = 10
BASE_CHANNELS = 96
NUM_CLASSES = 10
CLASS_DROPOUT_PROB = 0.1
NUM_EPOCHS = 5
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "Turn3_CFG"


def save_loss_graph(train_losses, validation_losses, path):
    """保存各 epoch 的平均训练/验证损失，无需打开绘图窗口。"""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    epochs = list(range(1, len(train_losses) + 1))
    figure = Figure(figsize=(8, 5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.plot(epochs, train_losses, marker="o", label="Train loss")
    axes.plot(epochs, validation_losses, marker="s", label="Validation loss")
    axes.set_title("Training and Validation Loss")
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Noise prediction MSE")
    axes.set_xticks(epochs)
    axes.set_ylim(bottom=0)
    axes.grid(alpha=0.3)
    axes.legend()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    figure.clear()


def train():
    torch.manual_seed(42)
    if __package__:
        from .data import train_loader, test_loader, DATA_ROOT
    else:
        from data import train_loader, test_loader, DATA_ROOT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = UNet(
        in_channels=1, out_channels=1, time_emb_dim=256, base_channels=BASE_CHANNELS,
        num_classes=NUM_CLASSES, class_dropout_prob=CLASS_DROPOUT_PROB,
    ).to(device)
    print(f"Model parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    weights_dir = OUTPUT_DIR / "weights"
    gifs_dir = OUTPUT_DIR / "gifs"
    weights_dir.mkdir(parents=True, exist_ok=True)
    gifs_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": "UNet", "in_channels": 1, "out_channels": 1,
        "time_emb_dim": 256, "base_channels": BASE_CHANNELS,
        "num_classes": NUM_CLASSES, "conditioning": "time_mlp(t) + label_embed(c)",
        "class_dropout_prob": CLASS_DROPOUT_PROB, "null_label": NUM_CLASSES,
        "label_dropout": "independent per sample during model.train(); disabled during eval",
        "num_epochs": NUM_EPOCHS, "batch_size": 64, "optimizer": "Adam",
        "learning_rate": 1e-4, "seed": 42, "num_timesteps": NUM_TIMESTEPS,
        "preview_every": PREVIEW_EVERY, "checkpoint_every": PREVIEW_EVERY,
        "gif_frame_every": GIF_FRAME_EVERY, "preview_sampler": "ddpm",
        "preview_cfg_scale": 1.0,
        "dataset": "MNIST", "data_root": str(DATA_ROOT), "normalization": "ToTensor [0, 1]",
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    num_epochs = NUM_EPOCHS
    global_step = 0
    train_losses = []
    validation_losses = []
    for epoch in range(num_epochs):
        total_train_loss = 0.0
        total_train_samples = 0
        for index, (x0, c) in enumerate(train_loader, start=1):
            x0 = x0.to(device)
            c = c.to(device)
            t = torch.randint(1, NUM_TIMESTEPS + 1, (x0.size(0),), device=device)
            real_eps = torch.randn_like(x0)
            xt = q_sample(x0, t, noise=real_eps)
            predicted_eps = model(xt, t, c)

            optimizer.zero_grad()
            loss = criterion(predicted_eps, real_eps)
            loss.backward()
            optimizer.step()
            global_step += 1
            loss_value = loss.item()
            total_train_loss += loss_value * x0.size(0)
            total_train_samples += x0.size(0)

            if index % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{index}/{len(train_loader)}], Loss: {loss_value:.4f}")

            if global_step % PREVIEW_EVERY == 0:
                name = f"epoch{epoch+1}_step{global_step}"
                torch.save(model.state_dict(), weights_dir / f"model_{name}.pth")
                gif_path = gifs_dir / f"denoise_{name}.gif"
                # 当前训练 batch 的第一张图，从它的随机 t 完整去噪到 0。
                # inference 内部关闭梯度，并在完成后恢复训练模式。
                save_inference_gif(
                    xt[:1], t[:1], model, gif_path, frame_every=GIF_FRAME_EVERY,
                    c=c[:1],
                )
                print(f"Denoising GIF saved: {gif_path}")

        # 每个 epoch 结束后验证完整测试集，按样本数汇总平均 loss。
        model.eval()
        total_test_loss = 0.0
        total_test_samples = 0
        with torch.no_grad():
            for test_x0, test_c in test_loader:
                test_x0 = test_x0.to(device)
                test_c = test_c.to(device)
                test_t = torch.randint(1, NUM_TIMESTEPS + 1, (test_x0.size(0),), device=device)
                test_eps = torch.randn_like(test_x0)
                test_xt = q_sample(test_x0, test_t, noise=test_eps)
                test_loss = criterion(model(test_xt, test_t, test_c), test_eps)
                total_test_loss += test_loss.item() * test_x0.size(0)
                total_test_samples += test_x0.size(0)
        model.train()
        train_loss = total_train_loss / total_train_samples
        validation_loss = total_test_loss / total_test_samples
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        with (OUTPUT_DIR / "loss.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "train_loss", "test_loss"])
            writer.writerows(zip(range(1, epoch + 2), train_losses, validation_losses))
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}")
        graph_path = OUTPUT_DIR / "graph" / "loss_curve.png"
        save_loss_graph(train_losses, validation_losses, graph_path)
        print(f"Loss graph saved: {graph_path}")

    # Preserve 100-update checkpoints and retain the final partial interval.
    final_path = weights_dir / "model_final.pth"
    torch.save(model.state_dict(), final_path)
    (OUTPUT_DIR / "completed.json").write_text(
        json.dumps({"epochs": num_epochs, "global_step": global_step,
                    "final_checkpoint": str(final_path)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Final checkpoint saved: {final_path}")


if __name__ == "__main__":
    train()
