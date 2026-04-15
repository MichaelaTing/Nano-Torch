"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


def _normalize_axes(axes, ndim):
    # 统一 axes 形式，并将负轴索引映射到正索引
    if axes is None:
        return tuple(range(ndim))
    if isinstance(axes, int):
        axes = (axes,)
    return tuple(axis if axis >= 0 else axis + ndim for axis in axes)


def _sum_array(a, axes=None, keepdims=False):
    if hasattr(array_api, "summation"):
        return array_api.summation(a, axis=axes, keepdims=keepdims)
    return array_api.sum(a, axis=axes, keepdims=keepdims)


def _empty_array(shape, dtype, device):
    if BACKEND == "nd":
        return array_api.empty(shape, dtype=dtype, device=device)
    return array_api.empty(shape, dtype=dtype)


def _full_array(shape, fill_value, dtype, device):
    if BACKEND == "nd":
        return array_api.full(shape, fill_value, dtype=dtype, device=device)
    return array_api.full(shape, fill_value, dtype=dtype)


def _compact_array(a):
    return a.compact() if hasattr(a, "compact") else a


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a**self.scalar

    def gradient(self, out_grad, node):
        return out_grad * self.scalar * (node.inputs[0] ** (self.scalar - 1))


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b


def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a: NDArray, b: NDArray):
        return a / b

    def gradient(self, out_grad: Tensor, node: Tensor):
        a, b = node.inputs
        return out_grad / b, -(out_grad * a) / (b * b)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a / self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ndim = len(a.shape)
        args_axes = list(range(ndim))
        if self.axes is None:
            # 与 numpy 一致：默认交换最后两个维度
            i, j = ndim - 2, ndim - 1
        else:
            i, j = self.axes
        args_axes[i], args_axes[j] = args_axes[j], args_axes[i]
        return array_api.transpose(a, tuple(args_axes))

    def gradient(self, out_grad, node):
        return transpose(out_grad, axes=self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad.reshape(node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        a_shape = node.inputs[0].shape
        out_shape = out_grad.shape
        # 先左侧补 1 做对齐，再把广播扩展过的维度求和还原
        aligned_shape = (1,) * (len(out_shape) - len(a_shape)) + a_shape
        reduce_axes = tuple(
            i
            for i, (old_dim, new_dim) in enumerate(zip(aligned_shape, out_shape))
            if old_dim != new_dim
        )
        reduced = summation(out_grad, axes=reduce_axes) if reduce_axes else out_grad
        return reshape(reduced, a_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None, keepdims=False):
        self.axes = None if axes is None else (axes,) if isinstance(axes, int) else axes
        self.keepdims = keepdims

    def compute(self, a: NDArray):
        axes = None if self.axes is None else _normalize_axes(self.axes, len(a.shape))
        return _sum_array(a, axes=axes, keepdims=self.keepdims)

    def gradient(self, out_grad: Tensor, node: Tensor):
        input_shape = node.inputs[0].shape
        axes = _normalize_axes(self.axes, len(input_shape)) if self.axes is not None else tuple(range(len(input_shape)))
        reshape_shape = [1 if i in axes else dim for i, dim in enumerate(input_shape)]
        return broadcast_to(reshape(out_grad, tuple(reshape_shape)), input_shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        if len(a.shape) > 2 or len(b.shape) > 2:
            if BACKEND == "nd":
                return NDArray(numpy.matmul(a.numpy(), b.numpy()), device=a.device)
            return numpy.matmul(a, b)
        return a @ b

    def gradient(self, out_grad: Tensor, node: Tensor):
        a, b = node.inputs
        ga = out_grad @ b.transpose()
        gb = a.transpose() @ out_grad

        # 先消去广播引入的多余前导 batch 维
        while len(ga.shape) > len(a.shape):
            ga = summation(ga, axes=0)
        while len(gb.shape) > len(b.shape):
            gb = summation(gb, axes=0)

        # 再对被广播的单例维求和，倒序处理可避免轴编号变化
        ga_reduce_axes = [i for i, (gdim, adim) in enumerate(zip(ga.shape, a.shape)) if adim == 1 and gdim != 1]
        for ax in reversed(ga_reduce_axes):
            ga = summation(ga, axes=ax)
        if ga.shape != a.shape:
            ga = reshape(ga, a.shape)

        gb_reduce_axes = [i for i, (gdim, bdim) in enumerate(zip(gb.shape, b.shape)) if bdim == 1 and gdim != 1]
        for ax in reversed(gb_reduce_axes):
            gb = summation(gb, axes=ax)
        if gb.shape != b.shape:
            gb = reshape(gb, b.shape)

        return ga, gb


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a: NDArray):
        return -a

    def gradient(self, out_grad: Tensor, node: Tensor):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a: NDArray):
        return array_api.log(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a: NDArray):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * node


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a: NDArray):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs[0].realize_cached_data()
        return out_grad * Tensor(a > 0, device=out_grad.device)


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        return array_api.tanh(a)

    def gradient(self, out_grad, node):
        x = node.inputs[0]
        y = exp(-x) + exp(x)
        z = mul_scalar(power_scalar(y, -2), 4)
        return out_grad * z


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        shape = args[0].shape
        new_shape = list(shape)
        new_shape.insert(self.axis, len(args))
        out = _empty_array(tuple(new_shape), dtype=args[0].dtype, device=getattr(args[0], "device", None))
        slices = []
        for i in range(len(new_shape)):
            if i != self.axis:
                slices.append(slice(new_shape[i]))
            else:
                slices.append(0)
        for i in range(len(args)):
            slices[self.axis] = i
            out[tuple(slices)] = args[i].reshape((1,) + shape)
        return out

    def gradient(self, out_grad, node):
        return (split(out_grad, self.axis),)


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        out_shape = [A.shape[i] for i in range(len(A.shape)) if i != self.axis]
        out_tensors = [_empty_array(tuple(out_shape), dtype=A.dtype, device=getattr(A, "device", None)) for i in range(A.shape[self.axis])]
        sl = []
        for i in range(len(A.shape)):
            if i == self.axis:
                sl.append(0)
            else:
                sl.append(slice(A.shape[i]))
        for i in range(len(out_tensors)):
            sl[self.axis] = i
            out_tensors[i] = _compact_array(array_api.reshape(_compact_array(A[tuple(sl)]), out_shape))
        return tuple(out_tensors)

    def gradient(self, out_grad, node):
        if isinstance(out_grad, Tensor):
            return (stack((out_grad,), self.axis),)
        return (stack(tuple(out_grad), self.axis),)


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.flip(a, self.axes)

    def gradient(self, out_grad, node):
        return flip(out_grad, self.axes)


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        new_shape = list(a.shape)
        for axis in self.axes:
            new_shape[axis] = a.shape[axis] * (self.dilation + 1)
        new_array = _full_array(tuple(new_shape), 0, dtype=a.dtype, device=getattr(a, "device", None))
        slices = [slice(0, shape) for shape in new_shape]
        for axis in self.axes:
            slices[axis] = slice(0, new_shape[axis], self.dilation + 1)
        new_array[tuple(slices)] = a
        return new_array

    def gradient(self, out_grad, node):
        return undilate(out_grad, self.axes, self.dilation)


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        slices = [slice(0, shape) for shape in a.shape]
        for axis in self.axes:
            slices[axis] = slice(0, a.shape[axis], self.dilation + 1)
        return a[tuple(slices)]

    def gradient(self, out_grad, node):
        return dilate(out_grad, self.axes, self.dilation)


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


def conv_im2col(Z, weight):
    N, H, W, C_in = Z.shape
    K, _, _, C_out = weight.shape
    Ns, Hs, Ws, Cs = Z.strides

    inner_dim = K * K * C_in
    A = numpy.lib.stride_tricks.as_strided(
        Z, shape=(N, H - K + 1, W - K + 1, K, K, C_in), strides=(Ns, Hs, Ws, Hs, Ws, Cs)
    ).reshape(-1, inner_dim)
    out = A @ weight.reshape(-1, C_out)
    return out.reshape(N, H - K + 1, W - K + 1, C_out)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        if hasattr(A, "pad") and hasattr(A, "as_strided"):
            A = A.pad(
                ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
            )
            N, H, W, C_in = A.shape
            K, _, _, C_out = B.shape
            Ns, Hs, Ws, Cs = A.strides
            inner_dim = K * K * C_in
            out_h = (H - K + 1) // self.stride
            out_w = (W - K + 1) // self.stride
            strided_A = _compact_array(
                A.as_strided(
                    shape=(N, out_h, out_w, K, K, C_in),
                    strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs),
                )
            ).reshape((-1, inner_dim))
            out = strided_A @ _compact_array(B).reshape((-1, C_out))
            return _compact_array(out).reshape((N, out_h, out_w, C_out))

        A_np = numpy.pad(
            A,
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
        )
        N, H, W, C_in = A_np.shape
        K, _, _, C_out = B.shape
        Ns, Hs, Ws, Cs = A_np.strides
        out_h = (H - K + 1) // self.stride
        out_w = (W - K + 1) // self.stride
        inner_dim = K * K * C_in
        strided_A = numpy.lib.stride_tricks.as_strided(
            A_np,
            shape=(N, out_h, out_w, K, K, C_in),
            strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs),
        ).reshape((-1, inner_dim))
        out = strided_A @ numpy.reshape(B, (-1, C_out))
        return out.reshape((N, out_h, out_w, C_out))

    def gradient(self, out_grad, node):
        # 输入张量形状为 (N, H, W, C_in)
        X = node.inputs[0]
        W = node.inputs[1]
        K = W.shape[0]
        if self.stride > 1:
            # 步长大于 1 时，先对输出梯度做膨胀再卷积
            out_grad = dilate(
                out_grad, (1, 2), self.stride - 1
            )  # 膨胀后恢复到稠密空间位置
        W_flip = flip(W, (0, 1))  # 卷积核在空间维翻转
        W_transpose = transpose(W_flip, (2, 3))  # 交换输入/输出通道位置
        X_grad = conv(out_grad, W_transpose, padding=K - 1 - self.padding)

        X_permute = transpose(X, (0, 3))
        out_grad_permute = transpose(transpose(out_grad, (0, 1)), (1, 2))
        W_grad_transpose = conv(X_permute, out_grad_permute, padding=self.padding)
        W_grad = transpose(transpose(W_grad_transpose, (0, 1)), (1, 2))
        return X_grad, W_grad


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
