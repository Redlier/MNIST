"""MNIST 数据读取、预处理和 mini-batch 工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np


@dataclass(frozen=True)
class DatasetSplit:
    """保存训练集和验证集，方便在训练脚本中传递。"""

    X: np.ndarray
    y: np.ndarray
    y_one_hot: np.ndarray


def one_hot_encode(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """把整数标签变成 one-hot 矩阵。

    例如标签 2 会变成 `[0, 0, 1, 0, ...]`。输出形状是 `(样本数, 类别数)`。
    """

    # 把输入统一成“一维整数标签”，这样后面的索引语句含义固定。
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("标签必须落在 [0, num_classes) 范围内")
    # 先创建全 0 表格，再把每个样本真实类别所在的格子改成 1。
    encoded = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


def _prepare_images(images: np.ndarray) -> np.ndarray:
    """把 `(N, 28, 28)` 图像拉平并归一化到 [0, 1]。"""

    # float32 足够表达像素，也比 float64 节省内存。
    images = np.asarray(images, dtype=np.float64)
    if images.ndim != 3:
        raise ValueError(f"MNIST 图像应是三维数组 (N, 28, 28)，实际为 {images.shape}")
    # 第一维 N 保留为样本数，-1 让 NumPy 自动计算 28*28=784。
    return (images.reshape(images.shape[0], -1) / 255.0).astype(np.float64)


def load_mnist(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """从 `mnist.npz` 读取并预处理训练集和测试集。

    返回 `X_train, y_train, X_test, y_test`，图像形状分别是 `(N, 784)`，
    标签形状是 `(N,)`。函数会拒绝缺少键或形状异常的文件。
    """

    # Path 能跨 Windows/Linux 组合路径，避免手写分隔符。
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path}。请先运行 download_data.py")
    # with 代码块结束后自动释放文件资源；allow_pickle=False 更安全。
    with np.load(path, allow_pickle=False) as data:
        required_keys = {"x_train", "y_train", "x_test", "y_test"}
        missing = required_keys.difference(data.files)
        if missing:
            raise ValueError(f"数据文件缺少键：{sorted(missing)}")
        X_train = _prepare_images(data["x_train"])
        X_test = _prepare_images(data["x_test"])
        y_train = np.asarray(data["y_train"], dtype=np.int64).reshape(-1)
        y_test = np.asarray(data["y_test"], dtype=np.int64).reshape(-1)
    if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]:
        raise ValueError("图像数量与标签数量不一致")
    return X_train, y_train, X_test, y_test


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[DatasetSplit, DatasetSplit]:
    """随机划分训练集和验证集，并同时生成 one-hot 标签。"""

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio 必须在 0 和 1 之间")
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X 与 y 的样本数不一致")
    # 固定 seed 后，同样的输入会得到同样的划分，便于复现实验。
    rng = np.random.default_rng(seed)
    indices = rng.permutation(X.shape[0])
    val_size = max(1, int(round(X.shape[0] * val_ratio)))
    # 前 val_size 个索引用于验证，其余索引用于训练。
    val_indices, train_indices = indices[:val_size], indices[val_size:]

    train = DatasetSplit(X[train_indices], y[train_indices], one_hot_encode(y[train_indices]))
    val = DatasetSplit(X[val_indices], y[val_indices], one_hot_encode(y[val_indices]))
    return train, val


def iterate_minibatches(
    X: np.ndarray,
    y_one_hot: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """逐批产生 `(X_batch, y_batch)`，最后一个不足整批的 batch 也保留。"""

    if batch_size <= 0:
        raise ValueError("batch_size 必须是正整数")
    if X.shape[0] != y_one_hot.shape[0]:
        raise ValueError("X 与 y_one_hot 的样本数不一致")
    # 先生成 0 到 N-1 的索引，而不是复制整个数据矩阵。
    indices = np.arange(X.shape[0])
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    # 每次只切出一个 batch；range 的步长就是 batch_size。
    for start in range(0, X.shape[0], batch_size):
        batch_indices = indices[start : start + batch_size]
        yield X[batch_indices], y_one_hot[batch_indices]
