"""NumPy 优化器。"""

from __future__ import annotations

import numpy as np


def sgd_update(
    parameters: dict[str, np.ndarray],
    gradients: dict[str, np.ndarray],
    learning_rate: float,
) -> None:
    """按 ``theta <- theta - learning_rate * dtheta`` 原地更新参数。"""

    # 学习率必须是有限的正浮点数；零或负数都不能产生正常的梯度下降。
    if not isinstance(learning_rate, (float, np.floating)):
        raise TypeError("learning_rate 必须是浮点数")
    learning_rate = float(learning_rate)
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate 必须是有限的正数")

    # 空字典无法完成参数更新，提前给出比后续 KeyError 更清晰的提示。
    if not parameters or not gradients:
        raise ValueError("parameters 和 gradients 都不能为空")

    parameter_keys = set(parameters.keys())
    gradient_keys = set(gradients.keys())
    expected_gradient_keys = {f"d{key}" for key in parameter_keys}
    missing_keys = expected_gradient_keys - gradient_keys
    unexpected_keys = gradient_keys - expected_gradient_keys
    if missing_keys or unexpected_keys:
        raise ValueError(
            "参数键与梯度键不匹配；"
            f"缺少梯度键：{sorted(missing_keys)}；"
            f"多余梯度键：{sorted(unexpected_keys)}"
        )

    # 逐个检查形状，再原地更新参数，保持调用者持有的数组引用不变。
    for key in parameter_keys:
        parameter = parameters[key]
        gradient = gradients[f"d{key}"]
        if not isinstance(parameter, np.ndarray) or not isinstance(gradient, np.ndarray):
            raise TypeError(f"参数 {key} 及其梯度必须是 NumPy 数组")
        if parameter.shape != gradient.shape:
            raise ValueError(
                f"参数 {key} 与梯度 d{key} 形状不一致："
                f"{parameter.shape} vs {gradient.shape}"
            )
        parameter -= learning_rate * gradient
