"""Optimization module"""

import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    # 执行一次参数更新
    def step(self):
        raise NotImplementedError()

    # 重置梯度
    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.u = {}

    def step(self):
        for i, param in enumerate(self.params):
            if i not in self.u:
                # 为每个参数维护独立的动量缓存
                self.u[i] = ndl.init.zeros(
                    *param.shape,
                    device=param.device,
                    dtype=param.dtype,
                    requires_grad=False,
                )
            if param.grad is None:
                continue
            # L2 正则等价于在梯度上加 weight_decay * w
            grad = param.grad.detach() + self.weight_decay * param.data
            # 指数滑动平均：u_t = m * u_{t-1} + (1-m) * g_t
            self.u[i] = self.momentum * self.u[i] + (1 - self.momentum) * grad
            param.data = param.data - self.u[i] * self.lr

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        raise NotImplementedError()


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.mu = {}
        self.v = {}

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if i not in self.mu:
                # 一阶矩估计（均值）
                self.mu[i] = ndl.init.zeros(
                    *param.shape,
                    device=param.device,
                    dtype=param.dtype,
                    requires_grad=False,
                )
                # 二阶矩估计（平方均值）
                self.v[i] = ndl.init.zeros(
                    *param.shape,
                    device=param.device,
                    dtype=param.dtype,
                    requires_grad=False,
                )
            if param.grad is None:
                continue
            grad = param.grad.detach() + param.data * self.weight_decay
            self.mu[i] = self.beta1 * self.mu[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            # 偏置修正，避免前几步矩估计系统性偏小
            m_hat = (self.mu[i]) / (1 - self.beta1**self.t)
            v_hat = (self.v[i]) / (1 - self.beta2**self.t)
            param.data = param.data - self.lr * m_hat / (v_hat**0.5 + self.eps)
