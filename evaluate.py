"""NumPy MLP 的评估工具（非核心，完整实现）。"""

from __future__ import annotations

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算分类准确率。"""

    # reshape(-1) 将列表、列向量等形式统一成一维标签序列。
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"真实标签与预测标签形状不一致：{y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("不能对空标签计算准确率")
    # 比较会得到 True/False 数组，mean 就是正确比例。
    return float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10
) -> np.ndarray:
    """构造混淆矩阵，行是真实类别，列是预测类别。"""

    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("真实标签与预测标签数量不一致")
    if np.any((y_true < 0) | (y_true >= num_classes)) or np.any(
        (y_pred < 0) | (y_pred >= num_classes)
    ):
        raise ValueError("标签超出类别范围")
    # 行表示真实类别，列表示预测类别，格子里累计样本数。
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def print_confusion_matrix(matrix: np.ndarray) -> None:
    """用简单的文本表格打印混淆矩阵。"""

    print("混淆矩阵（行=真实类别，列=预测类别）：")
    print(matrix)
