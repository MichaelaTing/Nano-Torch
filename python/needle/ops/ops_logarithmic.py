from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND


def _normalize_axes(axes, ndim):
    # 统一处理 None / int / 负轴索引
    if axes is None:
        return tuple(range(ndim))
    if isinstance(axes, int):
        axes = (axes,)
    return tuple(axis if axis >= 0 else axis + ndim for axis in axes)


def _sum_raw(x, axes=None, keepdims=False):
    if hasattr(array_api, "summation"):
        return array_api.summation(x, axis=axes, keepdims=keepdims)
    return array_api.sum(x, axis=axes, keepdims=keepdims)


def _broadcast_raw(x, shape):
    if hasattr(x, "broadcast_to"):
        return x.broadcast_to(shape)
    return array_api.broadcast_to(x, shape)


class LogSoftmax(TensorOp):
    def compute(self, Z):
        # 使用 shift 技巧避免 exp 溢出
        max_z = Z.max(axis=-1, keepdims=True)
        shifted = Z - _broadcast_raw(max_z, Z.shape)
        log_norm = array_api.log(_sum_raw(array_api.exp(shifted), axes=-1, keepdims=True))
        return shifted - _broadcast_raw(log_norm, Z.shape)

    def gradient(self, out_grad, node):
        y = node
        reduce_shape = list(y.shape)
        reduce_shape[-1] = 1
        summed = summation(out_grad, axes=-1)
        summed = broadcast_to(reshape(summed, tuple(reduce_shape)), y.shape)
        return out_grad - exp(y) * summed


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = None if axes is None else (axes,) if isinstance(axes, int) else axes

    def compute(self, Z):
        ndim = len(Z.shape)
        axes = _normalize_axes(self.axes, ndim)

        max_keep = Z
        for axis in axes:
            max_keep = max_keep.max(axis=axis, keepdims=True)
        shifted = Z - _broadcast_raw(max_keep, Z.shape)

        sum_exp = array_api.exp(shifted)
        for axis in sorted(axes, reverse=True):
            sum_exp = _sum_raw(sum_exp, axes=axis, keepdims=False)

        max_no_keep = Z
        for axis in sorted(axes, reverse=True):
            max_no_keep = max_no_keep.max(axis=axis, keepdims=False)

        return array_api.log(sum_exp) + max_no_keep

    def gradient(self, out_grad, node):
        Z = node.inputs[0]
        axes = _normalize_axes(self.axes, len(Z.shape))
        reshape_shape = [1 if i in axes else dim for i, dim in enumerate(Z.shape)]

        # 重新从输入计算稳定概率，避免数值很大时 logsumexp 舍入带来的精度损失
        max_keep = Z.realize_cached_data()
        for axis in axes:
            max_keep = max_keep.max(axis=axis, keepdims=True)
        max_keep = Tensor.make_const(max_keep)

        shifted = Z - broadcast_to(max_keep, Z.shape)
        exp_shifted = exp(shifted)
        denom = exp_shifted
        for axis in sorted(axes, reverse=True):
            denom = summation(denom, axes=axis)
        denom = broadcast_to(reshape(denom, tuple(reshape_shape)), Z.shape)
        probs = exp_shifted / denom

        out_grad = broadcast_to(reshape(out_grad, tuple(reshape_shape)), Z.shape)
        return out_grad * probs


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
