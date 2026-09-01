"""MLP 的交叉熵损失函数。"""

from __future__ import annotations

import numpy as np

import functions 


def cross_entropy_loss(probabilities: np.ndarray, targets_one_hot: np.ndarray) -> float:
    """计算 one-hot 标签下的平均交叉熵；输入形状均为 ``(B, C)``。"""
    probabilities=np.asarray(probabilities,dtype=np.float64)
    targets_one_hot=np.asarray(targets_one_hot,dtype=np.float64)
    if probabilities.ndim != 2 or targets_one_hot.ndim != 2:
        raise ValueError(
            "probabilities 和 targets_one_hot 都必须是二维数组 (B, C)；"
            f"实际形状为 {probabilities.shape} 和 {targets_one_hot.shape}"
        )
    if probabilities.shape != targets_one_hot.shape:
        raise ValueError(
            "probabilities 与 targets_one_hot 的形状必须完全一致；"
            f"实际为 {probabilities.shape} 和 {targets_one_hot.shape}"
        )
    batch_size = probabilities.shape[0]
    if batch_size == 0:
        raise ValueError("交叉熵不能处理空 batch")
    epsilon:float =1e-12
    safe_probabilities=probabilities+epsilon
    log_probabilities=np.log(safe_probabilities)
    one_hot_probabilities=log_probabilities*targets_one_hot
    fu_probabilities_sum=-np.sum(one_hot_probabilities,axis=1)
    loss=np.mean(fu_probabilities_sum)
    loss=float(loss)
    return loss


def softmax_cross_entropy_gradient(
    logits: np.ndarray, targets_one_hot: np.ndarray
) -> np.ndarray:
    """返回平均 softmax 交叉熵对 logits 的梯度 ``(softmax(logits)-y)/B``。"""
    logits=np.asarray(logits,dtype=np.float64)
    targets_one_hot=np.asarray(targets_one_hot,dtype=np.float64)
    if logits.ndim != 2 or targets_one_hot.ndim != 2:
        raise ValueError(
            "logits 和 targets_one_hot 都必须是二维数组 (B, C)；"
            f"实际形状为 {logits.shape} 和 {targets_one_hot.shape}"
        )
    if logits.shape != targets_one_hot.shape:
        raise ValueError(
            "logits 与 targets_one_hot 的形状必须完全一致；"
            f"实际为 {logits.shape} 和 {targets_one_hot.shape}"
        )
    batch_size=logits.shape[0]
    if batch_size == 0:
        raise ValueError("softmax+交叉熵梯度不能处理空 batch")
    P=functions.softmax(logits)
    dZ=(P-targets_one_hot)/batch_size
    return dZ
