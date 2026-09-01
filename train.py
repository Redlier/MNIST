"""MNIST NumPy MLP 的训练入口。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_loader import iterate_minibatches, load_mnist, train_val_split
from evaluate import accuracy_score
from losses import cross_entropy_loss
from model import MLP
from optim import sgd_update
from project_status import print_pending_implementations


@dataclass
class TrainConfig:
    """训练超参数。"""

    data_path: Path = Path("data/mnist.npz")
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 0.05
    val_ratio: float = 0.1
    seed: int = 42
    output_dir: Path = Path("outputs")


def train_model(
    model: MLP,
    X_train: np.ndarray,
    y_train_one_hot: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
) -> dict[str, list[float]]:
    """执行 epoch 训练、验证和日志记录。"""

    # 每个列表按 epoch 顺序保存一个标量，之后直接用于打印和画图。
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }
    y_train = np.argmax(y_train_one_hot, axis=1)
    for epoch in range(1, config.epochs + 1):
        batch_losses: list[float] = []
        batch_predictions: list[np.ndarray] = []
        # 训练集每次只取一个小批次，降低内存使用并增加更新次数。
        for X_batch, y_batch in iterate_minibatches(
            X_train, y_train_one_hot, config.batch_size, shuffle=True, seed=config.seed + epoch
        ):
            # logits 是未归一化的类别分数，形状为 (batch_size, 10)。
            logits = model.forward(X_batch)
            loss = cross_entropy_loss(model.predict_proba(X_batch), y_batch)
            # backward 使用刚才的 batch 计算每个参数的梯度。
            gradients = model.backward(X_batch, y_batch)
            sgd_update(model.params, gradients, config.learning_rate)
            batch_losses.append(loss)
            batch_predictions.append(np.argmax(logits, axis=1))

        # 把各 batch 的预测拼回完整训练集，计算本 epoch 准确率。
        train_predictions = np.concatenate(batch_predictions)
        train_loss = float(np.mean(batch_losses))
        train_accuracy = accuracy_score(y_train, train_predictions)
        val_accuracy = accuracy_score(y_val, model.predict(X_val))
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"loss={train_loss:.4f} | train_acc={train_accuracy:.4f} | val_acc={val_accuracy:.4f}"
        )
    return history


def save_history_plot(history: dict[str, list[float]], output_path: Path) -> None:
    """把训练损失和准确率曲线保存为 PNG。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], marker="o")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history["train_accuracy"], label="train")
    axes[1].plot(epochs, history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"训练曲线已保存：{output_path}")


def main() -> int:
    """命令行训练入口。"""

    parser = argparse.ArgumentParser(description="训练 NumPy MLP MNIST 分类器")
    parser.add_argument("--data", type=Path, default=Path("data/mnist.npz"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()
    config = TrainConfig(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    try:
        model = MLP([784, 256, 128, 10], seed=config.seed)
        X_train, y_train, X_test, y_test = load_mnist(config.data_path)
        train, val = train_val_split(X_train, y_train, config.val_ratio, config.seed)
        history = train_model(model, train.X, train.y_one_hot, val.X, val.y, config)
        print(f"测试集准确率：{accuracy_score(y_test, model.predict(X_test)):.4f}")
        save_history_plot(history, config.output_dir / "training_curves.png")
    except FileNotFoundError as error:
        print(f"数据尚未准备好：{error}")
        print("先运行：python download_data.py")
    except NotImplementedError:
        print_pending_implementations()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
