"""The module."""

from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
from functools import reduce
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""
    # Tensor的子类，用来指明这是一种特殊的张量


def _unpack_params(value: object) -> List[Tensor]:
    # 递归展开嵌套结构（Module / dict / list / tuple）中的所有参数
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        # 先收集自己，再继续向下遍历成员字段
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    # 指明如何获取某个给定模块的参数，通过递归方式实现
    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(
            self.__dict__
        )  # # self.__dict__是一个符号表，包含所有成员

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        # 递归切换到推理模式（影响 Dropout / BatchNorm 等行为）
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        # 递归切换到训练模式
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype)
        )
        self.bias = None
        if bias:
            self.bias = Parameter(
                init.kaiming_uniform(
                    out_features, 1, device=device, dtype=dtype
                ).transpose()
            )

    def forward(self, X: Tensor) -> Tensor:
        if len(X.shape) > 2:
            # 对高维输入先展平前导维，矩阵乘后再还原形状
            leading = X.shape[:-1]
            flat_n = reduce(lambda a, b: a * b, leading)
            output = X.reshape((flat_n, X.shape[-1])) @ self.weight
            output = output.reshape((*leading, self.out_features))
        else:
            output = X @ self.weight
        if self.bias:
            # 偏置按 batch 维广播到输出形状
            output = output + self.bias.broadcast_to(output.shape)
        return output


class Flatten(Module):
    def forward(self, X):
        flattened_dim = reduce(lambda a, b: a * b, X.shape[1:])
        return X.reshape((X.shape[0], flattened_dim))


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        assert len(logits.shape) == 2 and len(y.shape) == 1
        assert logits.shape[0] == y.shape[0]

        n, k = logits.shape
        # 使用 logsumexp 提升数值稳定性，避免直接 softmax 溢出
        log_sum_exp = ops.logsumexp(logits, axes=(1,))
        y_one_hot = init.one_hot(k, y, device=logits.device, dtype=logits.dtype)
        losses = log_sum_exp - (logits * y_one_hot).sum(axes=(1,))
        return losses.sum(axes=0) / n


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = np.float32(eps)
        self.momentum = np.float32(momentum)
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 2 and x.shape[1] == self.dim

        n = x.shape[0]
        w = ops.broadcast_to(self.weight, (n, self.dim))
        b = ops.broadcast_to(self.bias, (n, self.dim))

        if self.training:
            # 训练阶段用当前 batch 统计量，并更新 running 均值/方差
            mean = x.sum(axes=(0,)) / n
            one_minus_m = np.float32(1.0) - self.momentum
            mean_for_running = Tensor(
                mean.numpy(),
                device=self.running_mean.device,
                dtype=self.running_mean.dtype,
                requires_grad=False,
            )
            self.running_mean.data = (
                one_minus_m * self.running_mean + self.momentum * mean_for_running
            )

            mean_b = ops.broadcast_to(mean, (n, self.dim))
            var = ((x - mean_b) ** 2).sum(axes=(0,)) / n
            var_for_running = Tensor(
                var.numpy(),
                device=self.running_var.device,
                dtype=self.running_var.dtype,
                requires_grad=False,
            )
            self.running_var.data = (
                one_minus_m * self.running_var + self.momentum * var_for_running
            )

            std_b = ops.broadcast_to((var + self.eps) ** 0.5, (n, self.dim))
            out = (x - mean_b) / std_b
        else:
            # 推理阶段使用累计的 running 统计量
            mean_b = ops.broadcast_to(self.running_mean, (n, self.dim))
            std_b = ops.broadcast_to((self.running_var + self.eps) ** 0.5, (n, self.dim))
            out = (x - mean_b) / std_b
        return w * out + b


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2, 3)).transpose((1, 2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 2 and x.shape[1] == self.dim

        n = x.shape[0]
        w = ops.broadcast_to(self.weight, (n, self.dim))
        b = ops.broadcast_to(self.bias, (n, self.dim))

        mean = (x.sum(axes=(1,)) / self.dim).reshape((n, 1))
        mean_b = ops.broadcast_to(mean, (n, self.dim))
        var = ((x - mean_b) ** 2).sum(axes=(1,)) / self.dim
        std = ((var + self.eps) ** 0.5).reshape((n, 1))
        std_b = ops.broadcast_to(std, (n, self.dim))

        return w * ((x - mean_b) / std_b) + b


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x

        keep_prob = np.float32(1.0 - self.p)
        # 训练时按 keep_prob 采样，并用反向缩放保持期望不变
        mask = init.randb(*x.shape, p=keep_prob, device=x.device, dtype=x.dtype)
        return x * mask / keep_prob


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x
