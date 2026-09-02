# NumPy MLP for MNIST

深度学习实践微项目：使用纯 NumPy 从零实现多层感知机（MLP），完成 MNIST 手写数字分类。
项目不依赖 PyTorch、TensorFlow 或 scikit-learn 的模型与数据处理接口，只使用numpy手动实现所有核心功能。

- He/Xavier 参数初始化、ReLU、稳定 Softmax 和交叉熵损失
- 逐层反向传播与 mini-batch SGD 参数更新
- 使用中心差分法进行数值梯度检验
- 包含 MNIST 下载、预处理、训练、验证、混淆矩阵和训练曲线
- 使用类型标注、命令行参数和单元测试组织代码

## 网络结构

```
784 → 256 → 128 → 10
```

隐藏层使用 ReLU，输出层返回 logits，损失函数为 Softmax 交叉熵。

## 项目结构

| 文件 | 说明 |
| --- | --- |
| `download_data.py` | 从公开来源下载并校验 MNIST |
| `data_loader.py` | 数据读取、归一化、one-hot 编码和 mini-batch |
| `functions.py` | 初始化、ReLU 和 Softmax |
| `losses.py` | 交叉熵和合并梯度 |
| `model.py` | MLP 前向传播、反向传播和预测 |
| `optim.py` | SGD 参数更新 |
| `train.py` | 训练、验证和曲线保存 |
| `evaluate.py` | 准确率和混淆矩阵 |
| `gradcheck.py` | 中心差分梯度检验 |
| `tests/` | 自动化测试 |

## 环境与运行

```bash
python -m pip install -r requirements.txt
python download_data.py
python train.py --epochs 10 --batch-size 128 --learning-rate 0.05
```

训练曲线会保存到 `outputs/training_curves.png`。数据文件和训练输出已加入 `.gitignore`，不会提交到仓库。

运行测试：

```bash
python -m pytest tests -q
```

当前本地测试结果：

```text
5 passed
```

## 实验结果

实际数据实验结果：

| 指标 | 结果 |
| --- | --- |
| 测试集准确率 | 97.05% |
| 网络结构 | 784-256-128-10 |
| 优化器 | Mini-batch SGD |

训练数据来源：https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz

训练曲线截图：training_curves.png
