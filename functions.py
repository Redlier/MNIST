"""NumPy MLP 的基础数学函数。"""

from __future__ import annotations

import numpy as np


def initialize_parameters(
    layer_dims: list[int], method: str = "he", seed: int | None = 42
) -> dict[str, np.ndarray]:
    """按 He 或 Xavier 规则初始化各层参数，返回 ``W1,b1,...`` 字典。"""
    if not isinstance(layer_dims, (list, tuple)):
        raise TypeError("layer_dims 必须是由各层宽度组成的列表或元组")
    if len(layer_dims) < 2:
        raise ValueError("layer_dims 至少需要包含输入层和输出层两个宽度")
    for layer_index, value in enumerate(layer_dims):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value <= 0:
            raise ValueError(
                f"layer_dims[{layer_index}] 必须是正整数，实际为 {value!r}"
            )
    if not isinstance(method, str):
        raise TypeError("method 必须是字符串，可选值为 'he' 或 'xavier'")
    method_name = method.strip().lower()
    if method_name not in {"he", "xavier"}:
        raise ValueError(
            f"不支持的初始化方法 {method!r}；可选值为 'he' 或 'xavier'"
        )
    parameters: dict[str, np.ndarray] = {}
    rng=np.random.default_rng(seed)
    for layer_index in range(len(layer_dims)-1):
        n_prev=int(layer_dims[layer_index])
        n_curr=int(layer_dims[layer_index+1])
        weight_shape=(n_curr,n_prev)

        if method_name=="he":
            std=np.sqrt(2/n_prev)
        else:
            std=np.sqrt(1/n_prev)

        W=rng.normal(loc=0.0,scale=std,size=weight_shape).astype(np.float64)
        b=np.zeros(shape=(n_curr,),dtype=np.float64)

        parameters[f"W{layer_index+1}"]=W
        parameters[f"b{layer_index+1}"]=b

    return parameters


def relu(z: np.ndarray) -> np.ndarray:
    """逐元素计算 ``ReLU(z) = max(0, z)``，输出形状与输入相同。"""
    z=np.asarray(z,dtype=np.float64)
    a=np.maximum(0.0,z).astype(np.float64)
    return a


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """返回 ReLU 的 0/1 导数掩码，约定 ``z <= 0`` 时导数为 0。"""
    z=np.asarray(z,dtype=np.float64)
    positive_mask=z>0
    derivative=positive_mask.astype(np.float64)
    return derivative


def softmax(logits: np.ndarray) -> np.ndarray:
    """对二维 logits 按行计算数值稳定的 softmax 概率。"""
    logits=np.asarray(logits,dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(f"logits 必须是二维数组 (B, C)，实际形状为 {logits.shape}")
    row_max=np.max(logits,axis=1,keepdims=True)
    logits=logits-row_max
    exp_values=np.exp(logits)
    row_sum=exp_values.sum(axis=1,keepdims=True)
    probability=exp_values/row_sum
    return probability
