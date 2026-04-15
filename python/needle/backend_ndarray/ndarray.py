import operator
import math
from functools import reduce
import numpy as np
import os
from torch import set_float32_matmul_precision
from . import ndarray_backend_numpy
from . import ndarray_backend_cpu


def prod(x):
    # 连乘工具函数，等价于 numpy.prod 但适配 Python 元组
    return reduce(operator.mul, x, 1)


class BackendDevice:
    """A backend device, wrapps the implementation module."""

    def __init__(self, name, mod):
        self.name = name
        self.mod = mod

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return self.name + "()"

    def __getattr__(self, name):
        return getattr(self.mod, name)

    def enabled(self):
        return self.mod is not None

    def randn(self, *shape, dtype="float32"):
        # numpy 随机接口本身不直接带 dtype，这里统一转成目标类型
        return NDArray(np.random.randn(*shape).astype(dtype), device=self)

    def rand(self, *shape, dtype="float32"):
        # numpy 随机接口本身不直接带 dtype，这里统一转成目标类型
        return NDArray(np.random.rand(*shape).astype(dtype), device=self)

    def one_hot(self, n, i, dtype="float32"):
        return NDArray(np.eye(n, dtype=dtype)[i], device=self)

    def empty(self, shape, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        return NDArray.make(shape, device=self)

    def full(self, shape, fill_value, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        arr = self.empty(shape, dtype)
        arr.fill(fill_value)
        return arr


def cuda():
    """Return cuda device"""
    try:
        from . import ndarray_backend_cuda

        return BackendDevice("cuda", ndarray_backend_cuda)
    except ImportError:
        return BackendDevice("cuda", None)


def cpu_numpy():
    """Return numpy device"""
    return BackendDevice("cpu_numpy", ndarray_backend_numpy)


def cpu():
    """Return cpu device"""
    return BackendDevice("cpu", ndarray_backend_cpu)


def default_device():
    return cpu()


def all_devices():
    """return a list of all available devices"""
    return [cpu(), cuda(), cpu_numpy()]


class NDArrayMeta(type):
    def __instancecheck__(cls, instance):
        # np 后端下允许将 numpy.ndarray 视作 NDArray 参与类型判断
        if os.environ.get("NEEDLE_BACKEND", "nd") == "np" and isinstance(instance, np.ndarray):
            return True
        return super().__instancecheck__(instance)


class NDArray(metaclass=NDArrayMeta):
    """A generic ND array class that may contain multipe different backends
    i.e., a Numpy backend, a native CPU backend, or a GPU backend.

    This class will only contains those functions that you need to implement
    to actually get the desired functionality for the programming examples
    in the homework, and no more.

    For now, for simplicity the class only supports float32 types, though
    this can be extended if desired.
    """

    def __init__(self, other, device=None):
        """Create by copying another NDArray, or from numpy"""
        if isinstance(other, NDArray) and hasattr(other, "_handle"):
            # 从已有 NDArray 拷贝一份数据
            if device is None:
                device = other.device
            self._init(other.to(device) + 0.0)  # 这里会触发一次显式拷贝
        elif isinstance(other, np.ndarray):
            # 从 numpy 数组拷贝到底层后端存储
            device = device if device is not None else default_device()
            array = self.make(other.shape, device=device)
            array.device.from_numpy(np.ascontiguousarray(other), array._handle)
            self._init(array)
        else:
            # 其余输入先转成 numpy，再递归构造 NDArray
            array = NDArray(np.array(other), device=device)
            self._init(array)

    def _init(self, other):
        self._shape = other._shape
        self._strides = other._strides
        self._offset = other._offset
        self._device = other._device
        self._handle = other._handle

    @staticmethod
    def compact_strides(shape):
        """Utility function to compute compact strides"""
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):
            res.append(stride)
            stride *= shape[-i]
        return tuple(res[::-1])

    @staticmethod
    def make(shape, strides=None, device=None, handle=None, offset=0):
        """Create a new NDArray with the given properties.  This will allocation the
        memory if handle=None, otherwise it will use the handle of an existing
        array."""
        array = NDArray.__new__(NDArray)
        array._shape = tuple(shape)
        array._strides = NDArray.compact_strides(shape) if strides is None else strides
        array._offset = offset
        array._device = device if device is not None else default_device()
        if handle is None:
            array._handle = array.device.Array(prod(shape))
        else:
            array._handle = handle
        return array

    ### 属性与字符串表示
    @property
    def shape(self):
        return self._shape

    @property
    def strides(self):
        return self._strides

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        # 当前实现只支持 float32
        return "float32"

    @property
    def ndim(self):
        """Return number of dimensions."""
        return len(self._shape)

    @property
    def size(self):
        return prod(self._shape)

    def __repr__(self):
        return "NDArray(" + self.numpy().__str__() + f", device={self.device})"

    def __str__(self):
        return self.numpy().__str__()

    def __array__(self, dtype=None):
        arr = self.numpy()
        return arr.astype(dtype) if dtype is not None else arr

    def astype(self, dtype):
        return self.numpy().astype(dtype)

    ### 基础数组操作
    def fill(self, value):
        """Fill (in place) with a constant value."""
        self._device.fill(self._handle, value)

    def to(self, device):
        """Convert between devices, using to/from numpy calls as the unifying bridge."""
        if device == self.device:
            return self
        else:
            return NDArray(self.numpy(), device=device)

    def numpy(self):
        """convert to a numpy array"""
        return self.device.to_numpy(
            self._handle, self.shape, self.strides, self._offset
        )

    def is_compact(self):
        """Return true if array is compact in memory and internal size equals product
        of the shape dimensions"""
        return (
            self._strides == self.compact_strides(self._shape)
            and prod(self.shape) == self._handle.size
        )

    def compact(self):
        """Convert a matrix to be compact"""
        if self.is_compact():
            return self
        else:
            # 非连续视图先压实，便于后续高效计算
            out = NDArray.make(self.shape, device=self.device)
            self.device.compact(
                self._handle, out._handle, self.shape, self.strides, self._offset
            )
            return out

    def as_strided(self, shape, strides):
        """Restride the matrix without copying memory."""
        assert len(shape) == len(strides)
        return NDArray.make(
            shape, strides=strides, device=self.device, handle=self._handle
        )

    @property
    def flat(self):
        return self.reshape((self.size,))

    def reshape(self, new_shape):
        """
        Reshape the matrix without copying memory.  This will return a matrix
        that corresponds to a reshaped array but points to the same memory as
        the original array.

        Raises:
            ValueError if product of current shape is not equal to the product
            of the new shape, or if the matrix is not compact.

        Args:
            new_shape (tuple): new shape of the array

        Returns:
            NDArray : reshaped array; this will point to thep
        """

        new_shape = list(new_shape)
        if new_shape.count(-1) > 1:
            raise ValueError("Only one dimension can be inferred")
        if -1 in new_shape:
            known = prod([d for d in new_shape if d != -1])
            # 按元素总数自动推断 -1 维度
            inferred = prod(self.shape) // known
            new_shape[new_shape.index(-1)] = inferred
        new_shape = tuple(new_shape)

        assert prod(new_shape) == prod(self.shape), f"Mismatched number of elements. Before: {self.shape}, after: {new_shape}"
        if not self.is_compact():
            return self.compact().reshape(new_shape)

        return NDArray.make(
            new_shape,
            strides=NDArray.compact_strides(new_shape),
            device=self._device,
            handle=self._handle,
            offset=self._offset,
        )

    def permute(self, new_axes):
        """
        Permute order of the dimensions.  new_axes describes a permuation of the
        existing axes, so e.g.:
          - If we have an array with dimension "BHWC" then .permute((0,3,1,2))
            would convert this to "BCHW" order.
          - For a 2D array, .permute((1,0)) would transpose the array.
        Like reshape, this operation should not copy memory, but achieves the
        permuting by just adjusting the shape/strides of the array.  That is,
        it returns a new array that has the dimensions permuted as desired, but
        which points to the same memroy as the original array.

        Args:
            new_axes (tuple): permuation order of the dimensions

        Returns:
            NDarray : new NDArray object with permuted dimensions, pointing
            to the same memory as the original NDArray (i.e., just shape and
            strides changed).
        """

        assert len(new_axes) == len(self.shape), "Mismatched number of axes"

        new_shape = tuple(self.shape[axis] for axis in new_axes)
        new_strides = tuple(self.strides[axis] for axis in new_axes)

        return NDArray.make(
            new_shape,
            strides=new_strides,
            device=self._device,
            handle=self._handle,
            offset=self._offset,
        )

    def broadcast_to(self, new_shape):
        """
        Broadcast an array to a new shape.  new_shape's elements must be the
        same as the original shape, except for dimensions in the self where
        the size = 1 (which can then be broadcast to any size).  As with the
        previous calls, this will not copy memory, and just achieves
        broadcasting by manipulating the strides.

        Raises:
            assertion error if new_shape[i] != shape[i] for all i where
            shape[i] != 1

        Args:
            new_shape (tuple): shape to broadcast to

        Returns:
            NDArray: the new NDArray object with the new broadcast shape; should
            point to the same memory as the original array.
        """

        assert len(new_shape) >= len(self.shape), "Cannot broadcast to fewer dimensions"

        padded_shape = [1] * (len(new_shape) - len(self.shape)) + list(self.shape)
        padded_strides = [0] * (len(new_shape) - len(self.shape)) + list(self.strides)

        assert all(
            shape == new_shape[i] for i, shape in enumerate(padded_shape) if shape != 1
        ), "Mismatched broadcast dimension"

        for i in range(len(new_shape)):
            if padded_shape[i] != 1 or new_shape[i] == 1:
                continue
            # 通过 stride=0 实现广播维复用
            padded_strides[i] = 0

        return NDArray.make(
            new_shape,
            strides=tuple(padded_strides),
            device=self._device,
            handle=self._handle,
            offset=self._offset,
        )

    ### 元素读写
    def process_slice(self, sl, dim):
        """Convert a slice to an explicit start/stop/step"""
        start, stop, step = sl.start, sl.stop, sl.step
        if start == None:
            start = 0
        if start < 0:
            start = self.shape[dim]
        if stop == None:
            stop = self.shape[dim]
        if stop < 0:
            stop = self.shape[dim] + stop
        if step == None:
            step = 1

        # 这里不处理负步长这类更复杂的切片情况
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"
        return slice(start, stop, step)

    def __getitem__(self, idxs):
        """
        The __getitem__ operator in Python allows us to access elements of our
        array.  When passed notation such as a[1:5,:-1:2,4,:] etc, Python will
        convert this to a tuple of slices and integers (for singletons like the
        '4' in this example).  Slices can be a bit odd to work with (they have
        three elements .start .stop .step), which can be None or have negative
        entries, so for simplicity we wrote the code for you to convert these
        to always be a tuple of slices, one of each dimension.

        For this tuple of slices, return an array that subsets the desired
        elements.  As before, this can be done entirely through compute a new
        shape, stride, and offset for the new "view" into the original array,
        pointing to the same memory

        Raises:
            AssertionError if a slice has negative size or step, or if number
            of slices is not equal to the number of dimension (the stub code
            already raises all these errors.

        Args:
            idxs tuple: (after stub code processes), a tuple of slice elements
            coresponding to the subset of the matrix to get

        Returns:
            NDArray: a new NDArray object corresponding to the selected
            subset of elements.  As before, this should not copy memroy but just
            manipulate the shape/strides/offset of the new array, referecing
            the same array as the original one.
        """

        # 将单个索引统一包装成 tuple，便于后续按切片处理
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        idxs = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        assert len(idxs) == self.ndim, "Need indexes equal to number of dimensions"

        new_shape = tuple([(idx.stop - idx.start + idx.step - 1) // idx.step for idx in idxs])
        new_strides = tuple([stride * idx.step for stride, idx in zip(self.strides, idxs)])
        offset = sum([stride * idx.start for stride, idx in zip(self.strides, idxs)])
        return NDArray.make(
            new_shape, strides=new_strides, offset=offset, device=self.device, handle=self._handle
        )

    def __setitem__(self, idxs, other):
        """Set the values of a view into an array, using the same semantics
        as __getitem__()."""
        view = self.__getitem__(idxs)
        if isinstance(other, NDArray):
            assert prod(view.shape) == prod(other.shape)
            self.device.ewise_setitem(
                other.compact()._handle,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )
        else:
            self.device.scalar_setitem(
                prod(view.shape),
                other,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )

    ### 按元素与标量运算：加法、乘法、布尔比较等

    def ewise_or_scalar(self, other, ewise_func, scalar_func):
        """Run either an elementwise or scalar version of a function,
        depending on whether "other" is an NDArray or scalar
        """
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            ewise_func(self.compact()._handle, other.compact()._handle, out._handle)
        else:
            scalar_func(self.compact()._handle, other, out._handle)
        return out

    def __add__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_add, self.device.scalar_add
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_mul, self.device.scalar_mul
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_div, self.device.scalar_div
        )

    def __neg__(self):
        return self * (-1)

    def __pow__(self, other):
        if isinstance(other, NDArray):
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            return NDArray(np.power(self.numpy(), other.numpy()).astype(np.float32), device=self.device)

        # CUDA 的标量幂在负底数配合整数指数时可能不稳定，这里走精确整数幂路径以避免 NaN
        if isinstance(other, (int, np.integer)):
            e = int(other)
            if e == 0:
                out = NDArray.make(self.shape, device=self.device)
                out.fill(1.0)
                return out
            if e > 0:
                base = self
                result = None
                while e > 0:
                    if e & 1:
                        result = base if result is None else (result * base)
                    e >>= 1
                    if e:
                        base = base * base
                return result
        out = NDArray.make(self.shape, device=self.device)
        self.device.scalar_power(self.compact()._handle, other, out._handle)
        return out

    def maximum(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_maximum, self.device.scalar_maximum
        )

    ### 二元比较运算统一返回 0.0 / 1.0 的浮点结果
    def __eq__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_eq, self.device.scalar_eq)

    def __ge__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_ge, self.device.scalar_ge)

    def __ne__(self, other):
        return 1 - (self == other)

    def __gt__(self, other):
        return (self >= other) * (self != other)

    def __lt__(self, other):
        return 1 - (self >= other)

    def __le__(self, other):
        return 1 - (self > other)

    ### 按元素函数

    def log(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_log(self.compact()._handle, out._handle)
        return out

    def exp(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_exp(self.compact()._handle, out._handle)
        return out

    def tanh(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_tanh(self.compact()._handle, out._handle)
        return out

    ### 矩阵乘法
    def __matmul__(self, other):
        """Matrix multplication of two arrays.  This requires that both arrays
        be 2D (i.e., we don't handle batch matrix multiplication), and that the
        sizes match up properly for matrix multiplication.

        In the case of the CPU backend, you will implement an efficient "tiled"
        version of matrix multiplication for the case when all dimensions of
        the array are divisible by self.device.__tile_size__.  In this case,
        the code below will restride and compact the matrix into tiled form,
        and then pass to the relevant CPU backend.  For the CPU version we will
        just fall back to the naive CPU implementation if the array shape is not
        a multiple of the tile size

        The GPU (and numpy) versions don't have any tiled version (or rather,
        the GPU version will just work natively by tiling any input size).
        """

        assert self.ndim == 2 and other.ndim == 2
        assert self.shape[1] == other.shape[0]

        m, n, p = self.shape[0], self.shape[1], other.shape[1]

        # 如果矩阵尺寸对齐，则使用分块矩阵乘法加速
        if hasattr(self.device, "matmul_tiled") and all(
            d % self.device.__tile_size__ == 0 for d in (m, n, p)
        ):

            def tile(a, tile):
                return a.as_strided(
                    (a.shape[0] // tile, a.shape[1] // tile, tile, tile),
                    (a.shape[1] * tile, tile, a.shape[1], 1),
                )

            t = self.device.__tile_size__
            a = tile(self.compact(), t).compact()
            b = tile(other.compact(), t).compact()
            out = NDArray.make((a.shape[0], b.shape[1], t, t), device=self.device)
            self.device.matmul_tiled(a._handle, b._handle, out._handle, m, n, p)

            return (
                out.permute((0, 2, 1, 3))
                .compact()
                .reshape((self.shape[0], other.shape[1]))
            )

        else:
            out = NDArray.make((m, p), device=self.device)
            self.device.matmul(
                self.compact()._handle, other.compact()._handle, out._handle, m, n, p
            )
            return out

    ### 归约运算，例如整体或按某个轴求和/求最大值

    def reduce_view_out(self, axis, keepdims=False):
        """Return a view to the array set up for reduction functions and output array."""
        if isinstance(axis, tuple) and not axis:
            raise ValueError("Empty axis in reduce")

        if axis is None:
            view = self.compact().reshape((1,) * (self.ndim - 1) + (prod(self.shape),))
            out_shape = (1,) * self.ndim if keepdims else ()
            out = NDArray.make(out_shape, device=self.device)

        else:
            if isinstance(axis, (tuple, list)):
                assert len(axis) == 1, "Only support reduction over a single axis"
                axis = axis[0]
            if axis < 0:
                axis += self.ndim

            view = self.permute(
                tuple([a for a in range(self.ndim) if a != axis]) + (axis,)
            )
            out = NDArray.make(
                (
                    tuple([1 if i == axis else s for i, s in enumerate(self.shape)])
                    if keepdims
                    else tuple([s for i, s in enumerate(self.shape) if i != axis])
                ),
                device=self.device,
            )
        return view, out

    def sum(self, axis=None, keepdims=True):
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_sum(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def max(self, axis=None, keepdims=True):
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_max(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def flip(self, axes):
        """
        Flip this ndarray along the specified axes.
        Note: compact() before returning.
        """
        new_stride = list(self.strides)
        for i in axes:
            new_stride[i] = -new_stride[i]
        offset = np.sum([(self.shape[i] - 1) * self.strides[i] for i in axes])
        return NDArray.make(
            self.shape,
            strides=new_stride,
            device=self.device,
            handle=self._handle,
            offset=offset,
        ).compact()

    def pad(self, axes):
        """
        Pad this ndarray by zeros by the specified amount in `axes`,
        which lists for _all_ axes the left and right padding amount, e.g.,
        axes = ( (0, 0), (1, 1), (0, 0)) pads the middle axis with a 0 on the left and right side.
        """
        view_idxs = []
        new_shape = []
        for dim_size, (pad_before, pad_after) in zip(self._shape, list(axes)):
            new_shape.append(dim_size + pad_after + pad_before)
            view_idxs.append(slice(pad_before, dim_size + pad_after, 1))
        res = NDArray.make(new_shape, device=self.device)
        res.fill(0)
        res[tuple(view_idxs)] = self
        return res


def array(a, dtype="float32", device=None):
    """Convenience methods to match numpy a bit more closely."""
    device = device if device is not None else default_device()
    if dtype is None:
        arr = np.array(a)
        if arr.dtype == np.float64 and getattr(device, "name", "") == "cpu_numpy":
            return NDArray(arr.astype(np.float64, copy=False), device=device)
        return NDArray(arr.astype(np.float32, copy=False), device=device)
    if dtype in ("float32", np.float32):
        return NDArray(np.array(a, dtype=np.float32), device=device)
    if dtype in ("float64", np.float64) and getattr(device, "name", "") == "cpu_numpy":
        return NDArray(np.array(a, dtype=np.float64), device=device)
    raise AssertionError("Only float32 is supported on cpu/cuda devices")


def empty(shape, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.empty(shape, dtype)


def full(shape, fill_value, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.full(shape, fill_value, dtype)


def broadcast_to(array, new_shape):
    return array.broadcast_to(new_shape)


def reshape(array, new_shape):
    return array.reshape(new_shape)


def transpose(a, axes):
    return a.permute(axes)


def maximum(a, b):
    return a.maximum(b)


def log(a):
    return a.log()


def exp(a):
    return a.exp()


def tanh(a):
    return a.tanh()


def summation(a, axis=None, keepdims=False):
    if axis is None:
        return a.sum(axis=None, keepdims=keepdims)

    if isinstance(axis, int):
        axis = (axis,)
    else:
        axis = tuple(axis)

    ndim = len(a.shape)
    axis = tuple(ax if ax >= 0 else ax + ndim for ax in axis)

    out = a
    if keepdims:
        for ax in axis:
            out = out.sum(axis=ax, keepdims=True)
    else:
        for ax in sorted(axis, reverse=True):
            out = out.sum(axis=ax, keepdims=False)
    return out


def flip(a, axes):
    return a.flip(axes)
