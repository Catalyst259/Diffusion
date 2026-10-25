"""Brief installation/GPU verification; run after the installer has exited."""
import datetime as dt
import io
import json
from pathlib import Path
import subprocess
import sys
import traceback


def main():
    result = {
        "time": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version,
        "executable": sys.executable,
        "status": "failed",
    }
    destination = Path(__file__).resolve().parent / "gpu-smoke-result.json"

    def stage(name):
        result["last_stage"] = name
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("GPU smoke stage:", name, flush=True)

    try:
        stage("imports")
        import torch
        import torchvision
        from PIL import Image
        from torchvision.ops import nms

        result.update(torch=torch.__version__, torchvision=torchvision.__version__,
                      cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version())
        assert sys.version_info[:2] == (3, 12), "Expected Python 3.12"
        assert torch.__version__ == "2.11.0+cu128", "Unexpected PyTorch build"
        assert torchvision.__version__ == "0.26.0+cu128", "Unexpected torchvision build"
        assert torch.cuda.is_available(), "CUDA is unavailable"
        result["gpu"] = torch.cuda.get_device_name(0)
        result["compute_capability"] = torch.cuda.get_device_capability(0)

        stage("CUDA matrix and training")
        torch.manual_seed(42)
        device = torch.device("cuda")
        matrix = torch.randn(128, 128, device=device)
        product = matrix @ matrix.T
        assert torch.isfinite(product).all().item(), "GPU matrix multiplication failed"
        result["matrix_test"] = "passed"

        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 4)),
            torch.nn.Flatten(),
            torch.nn.Linear(128, 10),
        ).to(device)
        inputs = torch.randn(8, 3, 32, 32, device=device)
        labels = torch.randint(0, 10, (8,), device=device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        losses = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(inputs), labels)
            assert torch.isfinite(loss).item(), "Non-finite training loss"
            loss.backward()
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
                       for parameter in model.parameters()), "Invalid gradients"
            optimizer.step()
            losses.append(loss.item())
        assert any(not torch.equal(old, parameter.detach())
                   for old, parameter in zip(before, model.parameters())), "Optimizer did not update weights"
        result["training_losses"] = losses
        result["forward_backward_optimizer"] = "passed"

        boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]],
                             device=device, dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8, 0.7], device=device)
        assert nms(boxes, scores, 0.5).cpu().tolist() == [0, 2], "torchvision CUDA NMS failed"
        torch.cuda.synchronize()
        result["torchvision_cuda_nms"] = "passed"

        # Check the image/plotting packages moved to pip to resolve the DLL conflict.
        stage("image codecs")
        assert Path(Image.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        result["pillow"] = Image.__version__
        for image_format in ("PNG", "JPEG", "TIFF"):
            buffer = io.BytesIO()
            Image.new("RGB", (8, 8), color="blue").save(buffer, format=image_format)
            buffer.seek(0)
            with Image.open(buffer) as restored:
                restored.load()
                assert restored.size == (8, 8)
        result["image_roundtrips"] = "PNG, JPEG, TIFF passed"

        stage("CPU numerical backend")
        import numpy as np
        import scipy.linalg
        from sklearn.linear_model import LinearRegression
        from threadpoolctl import threadpool_info
        rng = np.random.default_rng(42)
        design = rng.standard_normal((64, 2))
        target = design @ np.array([1.0, 2.0]) + 0.5
        fitted = LinearRegression().fit(design, target)
        assert np.allclose(fitted.coef_, [1.0, 2.0])
        positive = design.T @ design + np.eye(2)
        solution = scipy.linalg.solve(positive, np.ones(2), assume_a="pos")
        assert np.allclose(positive @ solution, np.ones(2))
        pools = threadpool_info()
        assert any(pool.get("internal_api") == "openblas" for pool in pools)
        result["cpu_numerical_backend"] = "NumPy, SciPy, scikit-learn passed with OpenBLAS"
        result["threadpool_libraries"] = [
            {key: pool.get(key) for key in ("internal_api", "prefix", "filepath", "version", "threading_layer")}
            for pool in pools
        ]

        stage("plotting imports")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        stage("plot rendering")
        artifacts = Path(__file__).resolve().parent / "gpu-smoke-artifacts"
        artifacts.mkdir(exist_ok=True)
        figure, axes = plt.subplots()
        sns.lineplot(x=[1, 2, 3], y=losses, ax=axes)
        figure.savefig(artifacts / "training-loss.png")
        plt.close(figure)
        result["plotting"] = "passed"

        stage("TensorBoard")
        from torch.utils.tensorboard import SummaryWriter
        with SummaryWriter(log_dir=str(artifacts / "tensorboard")) as writer:
            for step, loss_value in enumerate(losses):
                writer.add_scalar("smoke/loss", loss_value, step)
        result["tensorboard"] = "passed"

        stage("dependency consistency")
        check = subprocess.run([sys.executable, "-m", "pip", "check"],
                               capture_output=True, text=True, errors="replace")
        result["pip_check"] = (check.stdout + check.stderr).strip()
        assert check.returncode == 0, "pip check found incompatible dependencies"
        result["status"] = "passed"
    except Exception:
        result["error"] = traceback.format_exc()
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
