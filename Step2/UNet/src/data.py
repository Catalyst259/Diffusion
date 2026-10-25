import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pathlib import Path
from torch.utils.data import DataLoader
# 采用 MNIST 数据集
__all__ = ["DATA_ROOT", "train_dataset", "train_loader", "test_dataset", "test_loader"]
DATA_ROOT = Path(__file__).resolve().parent / "data"
train_dataset = datasets.MNIST(
    root=DATA_ROOT,
    train=True,
    download=False,
    transform=transforms.ToTensor()
)

test_dataset = datasets.MNIST(
    root=DATA_ROOT,
    train=False,
    download=False,
    transform=transforms.ToTensor()
)
# 训练时加载
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)
# 测试时加载
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False
)
if __name__ == "__main__":
    # 测试数据加载器
    for images, labels in train_loader:
        print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
        break
    for images, labels in test_loader:
        print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
        break
