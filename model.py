"""纯 NumPy 实现的多层感知机分类器。"""

from __future__ import annotations

import numpy as np

import functions
import losses


class MLP:
    """默认结构为 ``784 -> 256 -> 128 -> 10`` 的全连接分类器。"""

    def __init__(
        self,
        layer_dims: list[int] | None = None,
        initialization: str = "he",
        seed: int | None = 42,
    ) -> None:
        """保存网络结构，并初始化参数与前向缓存。"""
        if layer_dims is None:
            layer_dims=[784,256,128,10]
        self.layer_dims=list(layer_dims)
        self.num_layers=len(layer_dims)-1
        self.initialization=initialization
        self.seed=seed
        self.params=functions.initialize_parameters(layer_dims,initialization,seed)
        self.cache=dict()


    def forward(self, X: np.ndarray) -> np.ndarray:
        """执行批量前向传播，返回未归一化的 logits。"""
        X=np.asarray(X,dtype=np.float64)
        if X.ndim!=2:
            raise ValueError(f"X 必须是二维数组 (B, D)，实际形状为 {X.shape}")
        if X.shape[1]!=self.layer_dims[0]:
            raise ValueError(
                "X 的特征数必须等于网络输入层宽度；"
                f"期望 {self.layer_dims[0]}，实际为 {X.shape[1]}"
            )
        #X=np.asarray(X,dtype=np.float32)
        self.cache.clear()
        self.cache["A0"]=X
        for layer_index in range(self.num_layers):
            Z = self.cache[f"A{layer_index}"] @ self.params[f"W{layer_index+1}"].T + self.params[f"b{layer_index+1}"]
            self.cache[f"Z{layer_index+1}"]=Z
            if layer_index+1 < self.num_layers:
                self.cache[f"A{layer_index+1}"]=functions.relu(self.cache[f"Z{layer_index+1}"])
            else:
                self.logits = self.cache[f"Z{layer_index+1}"]
                return self.logits

    def backward(self, X: np.ndarray, targets_one_hot: np.ndarray) -> dict[str, np.ndarray]:
        """计算一个 batch 对全部参数的梯度，返回 ``dW1, db1, ...``。"""
        X=np.asarray(X,dtype=np.float64)
        targets_one_hot=np.asarray(targets_one_hot,dtype=np.float64)
        if not (X.ndim ==2 and targets_one_hot.ndim==2):
            raise ValueError(
                "X 和 targets_one_hot 都必须是二维数组；"
                f"实际形状为 {X.shape} 和 {targets_one_hot.shape}"
            )
        if X.shape[0] != targets_one_hot.shape[0]:
            raise ValueError(
                "X 与 targets_one_hot 的 batch 大小必须一致；"
                f"实际为 {X.shape[0]} 和 {targets_one_hot.shape[0]}"
            )
        if not (X.shape[1]==self.layer_dims[0] and targets_one_hot.shape[1]==self.layer_dims[-1]):
            raise ValueError(
                "X 或 targets_one_hot 的列数与网络结构不匹配；"
                f"X 应为 {self.layer_dims[0]} 列、标签应为 {self.layer_dims[-1]} 列，"
                f"实际为 {X.shape[1]} 和 {targets_one_hot.shape[1]}"
            )
        if X.shape[0] == 0:
            raise ValueError("backward 不能处理空 batch")
        self.forward(X)
        dZ_L = losses.softmax_cross_entropy_gradient(self.logits,targets_one_hot)
        self.gradients=dict()
        self.cache[f"dZ{self.num_layers}"]=dZ_L
        for layer_index in range(self.num_layers,0,-1):
            dW = self.cache[f"dZ{layer_index}"].T @ self.cache[f"A{layer_index-1}"]
            db = np.sum(self.cache[f"dZ{layer_index}"],axis=0)
            self.gradients[f"dW{layer_index}"]=dW
            self.gradients[f"db{layer_index}"]=db
            self.cache[f"dA{layer_index-1}"]= self.cache[f"dZ{layer_index}"] @ self.params[f"W{layer_index}"]
            if layer_index-1>0:
                self.cache[f"dZ{layer_index-1}"]=self.cache[f"dA{layer_index-1}"] * functions.relu_derivative(self.cache[f"Z{layer_index-1}"])
        return self.gradients


    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回每个样本的类别概率。"""

        from functions import softmax

        return softmax(self.forward(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """返回每个样本的预测类别编号。"""

        return np.argmax(self.predict_proba(X), axis=1)
