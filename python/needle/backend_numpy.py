"""This file defies specific implementations of devices when using numpy as NDArray backend."""

import numpy


class NumpyArray(numpy.ndarray):
    """NumPy array subclass with an NDArray-like .numpy() API."""

    def numpy(self):
        return numpy.asarray(self)


def as_numpy_array(x, dtype=None):
    # 统一转换为带 .numpy() 接口的 NumpyArray 子类
    arr = numpy.array(x, dtype=dtype)
    if isinstance(arr, NumpyArray):
        return arr
    return arr.view(NumpyArray)


class Device:
    """Baseclass of all device"""


class CPUDevice(Device):
    """Represents data that sits in CPU"""

    def __repr__(self):
        return "cpu()"

    def __hash__(self):
        return self.__repr__().__hash__()

    def __eq__(self, other):
        return isinstance(other, CPUDevice)

    def enabled(self):
        return True

    def zeros(self, *shape, dtype="float32"):
        return as_numpy_array(numpy.zeros(shape, dtype=dtype))

    def ones(self, *shape, dtype="float32"):
        return as_numpy_array(numpy.ones(shape, dtype=dtype))

    def randn(self, *shape):
        # numpy 随机接口本身不直接带 dtype，这里统一转成 float32
        return as_numpy_array(numpy.random.randn(*shape).astype("float32"))

    def rand(self, *shape):
        # numpy 随机接口本身不直接带 dtype，这里统一转成 float32
        return as_numpy_array(numpy.random.rand(*shape).astype("float32"))

    def one_hot(self, n, i, dtype="float32"):
        return as_numpy_array(numpy.eye(n, dtype=dtype)[i])

    def empty(self, shape, dtype="float32"):
        return as_numpy_array(numpy.empty(shape, dtype=dtype))

    def full(self, shape, fill_value, dtype="float32"):
        return as_numpy_array(numpy.full(shape, fill_value, dtype=dtype))


class CUDADevice(Device):
    """Stub CUDA device for numpy backend; always reports unavailable."""

    def __repr__(self):
        return "cuda()"

    def __hash__(self):
        return self.__repr__().__hash__()

    def __eq__(self, other):
        return isinstance(other, CUDADevice)

    def enabled(self):
        return False


def cpu():
    """Return cpu device"""
    return CPUDevice()


def cuda():
    """Return CUDA stub device (unavailable under numpy backend)."""
    # numpy backend 下仅保留占位设备，避免上层分支判断复杂化
    return CUDADevice()


def default_device():
    return cpu()


def all_devices():
    """return a list of all available devices"""
    return [cpu(), cuda()]
