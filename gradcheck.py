"""中心差分数值梯度检验工具。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


ArrayLoss = Callable[[np.ndarray], float]


def central_difference_gradient(
    loss_function: ArrayLoss, parameter: np.ndarray, epsilon: float = 1e-5
) -> np.ndarray:
    """逐元素使用中心差分近似梯度。

    对第 i 个元素，近似公式是
    `dL/dtheta_i ≈ [L(theta_i+epsilon)-L(theta_i-epsilon)]/(2*epsilon)`。
    参数可以是任意形状，输出形状与 parameter 完全相同。函数会复制数组，
    不会永久修改调用者的参数。
    """

    if epsilon <= 0:
        raise ValueError("epsilon 必须为正数")
    parameter = np.asarray(parameter, dtype=np.float64)
    # 创建与参数同形状的数组接收每个元素的数值梯度。
    numerical = np.zeros_like(parameter, dtype=np.float64)
    for index in np.ndindex(parameter.shape):
        # 暂存当前值，函数结束前恢复，保证调用者看到的参数不变。
        original = parameter[index]
        parameter[index] = original + epsilon
        loss_plus = float(loss_function(parameter))
        parameter[index] = original - epsilon
        loss_minus = float(loss_function(parameter))
        parameter[index] = original
        numerical[index] = (loss_plus - loss_minus) / (2.0 * epsilon)
    return numerical


def relative_error(analytic: np.ndarray, numerical: np.ndarray) -> float:
    """计算常用的相对误差 `max|a-n| / max(1e-12, max|a|+|n|)`。"""

    analytic = np.asarray(analytic, dtype=np.float64)
    numerical = np.asarray(numerical, dtype=np.float64)
    if analytic.shape != numerical.shape:
        raise ValueError("解析梯度与数值梯度形状不一致")
    denominator = max(1e-12, float(np.max(np.abs(analytic) + np.abs(numerical))))
    return float(np.max(np.abs(analytic - numerical)) / denominator)


def check_parameter_gradient(
    loss_function: ArrayLoss,
    parameter: np.ndarray,
    analytic_gradient: np.ndarray,
    epsilon: float = 1e-5,
    tolerance: float = 1e-6,
) -> bool:
    """打印一个参数的数值/解析梯度相对误差并返回是否通过。"""

    numerical = central_difference_gradient(loss_function, parameter, epsilon)
    error = relative_error(analytic_gradient, numerical)
    passed = error < tolerance
    print(f"梯度检验：relative_error={error:.3e}，阈值={tolerance:.1e}，结果={'通过' if passed else '未通过'}")
    return passed
