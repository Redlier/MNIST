"""项目一的可运行自测。

核心尚未填写时，相关测试会显示为 skipped，并明确列出函数名；填写后无需修改测试。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loader import iterate_minibatches, one_hot_encode  # noqa: E402
from evaluate import accuracy_score, confusion_matrix  # noqa: E402
from gradcheck import check_parameter_gradient  # noqa: E402
from losses import cross_entropy_loss  # noqa: E402
from model import MLP  # noqa: E402
from optim import sgd_update  # noqa: E402


class ScaffoldTests(unittest.TestCase):
    def test_data_shapes(self) -> None:
        X = np.zeros((7, 784), dtype=np.float32)
        y = np.arange(7) % 10
        encoded = one_hot_encode(y)
        batches = list(iterate_minibatches(X, encoded, batch_size=3, shuffle=False))
        self.assertEqual(encoded.shape, (7, 10))
        self.assertEqual([batch[0].shape[0] for batch in batches], [3, 3, 1])

    def test_evaluation_helpers(self) -> None:
        truth = np.array([0, 1, 1, 2])
        prediction = np.array([0, 1, 2, 2])
        self.assertAlmostEqual(accuracy_score(truth, prediction), 0.75)
        self.assertEqual(confusion_matrix(truth, prediction, 3).sum(), 4)

    def test_model_shape_after_implementation(self) -> None:
        try:
            model = MLP([4, 5, 3], seed=1)
            logits = model.forward(np.zeros((2, 4), dtype=np.float32))
        except NotImplementedError as error:
            self.skipTest(f"核心函数待实现：{error}")
        self.assertEqual(logits.shape, (2, 3))
        # 模型契约要求缓存每一层，便于反向传播和初学者排查形状。
        cache = getattr(model, "cache", {})
        expected_shapes = {
            "A0": (2, 4),
            "Z1": (2, 5),
            "A1": (2, 5),
            "Z2": (2, 3),
        }
        for cache_name, expected_shape in expected_shapes.items():
            self.assertIn(cache_name, cache, f"前向缓存缺少 {cache_name}")
            self.assertEqual(tuple(cache[cache_name].shape), expected_shape)

    def test_gradient_check_after_implementation(self) -> None:
        try:
            model = MLP([4, 5, 3], seed=1)
            X = np.array([[1.0, 0.0, -1.0, 0.5], [0.2, -0.3, 0.7, 0.1]], dtype=np.float64)
            targets = one_hot_encode(np.array([0, 2]), 3)
            model.forward(X)
            gradients = model.backward(X, targets)

            # 只对一个小参数矩阵做中心差分，避免教学测试运行太慢。
            original_parameter = model.params["W1"]

            def loss_for_candidate(candidate: np.ndarray) -> float:
                old_parameter = model.params["W1"]
                model.params["W1"] = candidate
                try:
                    probabilities = model.predict_proba(X)
                    return cross_entropy_loss(probabilities, targets)
                finally:
                    model.params["W1"] = old_parameter

            passed = check_parameter_gradient(
                loss_for_candidate,
                original_parameter.copy(),
                gradients["dW1"],
                epsilon=1e-3,
                tolerance=1e-5,
            )
        except NotImplementedError as error:
            self.skipTest(f"核心函数待实现：{error}")
        self.assertTrue(passed)

    def test_overfit_small_batch_after_implementation(self) -> None:
        """填写核心后，这个测试应能把很小的人工数据集拟合到较高准确率。"""

        try:
            rng = np.random.default_rng(7)
            X = rng.normal(size=(100, 4)).astype(np.float64)
            # 标签由一个简单线性规则生成，网络应能在小数据集上记住它。
            labels = (X[:, 0] + 0.7 * X[:, 1] > 0).astype(np.int64)
            targets = one_hot_encode(labels, 2)
            model = MLP([4, 8, 2], seed=1)
            for _ in range(300):
                model.forward(X)
                gradients = model.backward(X, targets)
                sgd_update(model.params, gradients, learning_rate=0.1)
            predictions = model.predict(X)
        except NotImplementedError as error:
            self.skipTest(f"核心函数待实现：{error}")
        self.assertGreaterEqual(accuracy_score(labels, predictions), 0.95)


if __name__ == "__main__":
    unittest.main(verbosity=2)
