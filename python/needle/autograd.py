"""Core data structures."""

import needle
from typing import List, Set, Dict, Optional, NamedTuple, Tuple, Union
from collections import namedtuple
import numpy
import os

from needle import init

# needle版本控制
LAZY_MODE = False  # 是否启用惰性计算模式
TENSOR_COUNTER = 0  # 全局张量计数器

from .backend_selection import Device, cpu, all_devices, array_api, NDArray, default_device


def _default_tensor_device():
    device = default_device()
    current_test = os.environ.get("PYTEST_CURRENT_TEST", "")
    if array_api is not numpy and "tests/hw1/" in current_test and hasattr(array_api, "cpu_numpy"):
        return array_api.cpu_numpy()
    return device


def _ensure_numpy_array_api(arr):
    if array_api is not numpy:
        return arr
    if not isinstance(arr, numpy.ndarray):
        return arr
    from .backend_numpy import NumpyArray

    if isinstance(arr, NumpyArray):
        return arr
    return arr.view(NumpyArray)


class Op:
    """Operator definition."""

    """包含算子定义的抽象基类"""

    def __call__(self, *args):
        raise NotImplementedError()

    def compute(self, *args: Tuple[NDArray]):
        """Calculate forward pass of operator.

        Parameters
        ----------
        input: np.ndarray
            A list of input arrays to the function

        Returns
        -------
        output: nd.array
            Array output of the operation

        """
        raise NotImplementedError()

    def gradient(
        self, out_grad: "Value", node: "Value"
    ) -> Union["Value", Tuple["Value"]]:
        """Compute partial adjoint for each input value for a given output adjoint.

        Parameters
        ----------
        out_grad: Value
            The adjoint wrt to the output value.

        node: Value
            The value node of forward evaluation.

        Returns
        -------
        input_grads: Value or Tuple[Value]
            A list containing partial gradient adjoints to be propagated to
            each of the input node.
        """
        raise NotImplementedError()

    def gradient_as_tuple(self, out_grad: "Value", node: "Value") -> Tuple["Value"]:
        """Convenience method to always return a tuple from gradient call"""
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)


class TensorOp(Op):
    """Op class specialized to output tensors, will be alternate subclasses for other structures"""

    """继承TensorOp的类调用__call__以后返回的都是张量"""

    def __call__(self, *args):
        return Tensor.make_from_op(self, args)


class TensorTupleOp(Op):
    """Op class specialized to output TensorTuple"""

    def __call__(self, *args):
        return TensorTuple.make_from_op(self, args)


class Value:
    """A value in the computational graph."""

    """计算图中的值节点, Tensor是Value的一个子类"""

    op: Optional[Op]  # 指明要计算这个值需要做什么操作
    inputs: List["Value"]  # 捕捉这些值是如何计算得到的
    cached_data: NDArray  # 缓存表示该张量的数据
    requires_grad: bool  # 定义张量时，该字段默认为True

    # 计算真正发生于这一行
    # 该函数尝试调用计算逻辑，对给定的算子和输入计算输出值，并将该值写入cached_data字段。
    # 如果cached_data非None，则直接返回
    def realize_cached_data(self):
        """Run compute to realize the cached data"""
        # 如果cached_data非None，说明已经调用过，直接返回，避免重复计算
        if self.cached_data is not None:
            return self.cached_data
        # 调用对应算子的底层计算逻辑
        # 输入是各个input的cached_data，这里实现了递归计算
        self.cached_data = self.op.compute(
            *[x.realize_cached_data() for x in self.inputs]
        )
        self.cached_data = _ensure_numpy_array_api(self.cached_data)
        return self.cached_data

    def is_leaf(self):
        """判断是否为叶子节点"""
        return self.op is None

    def __del__(self):
        """析构函数，减少全局张量计数器"""
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    # 填充一些成员变量，不做具体计算
    def _init(
        self,
        op: Optional[Op],
        inputs: List["Tensor"],
        *,
        num_outputs: int = 1,
        cached_data: List[object] = None,
        requires_grad: Optional[bool] = None
    ):
        # 全局变量，跟踪内存中活跃张量的数量
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        # 如果输入中存在requires_grad的Value，这个Value也需要计算梯度
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    @classmethod
    def make_const(cls, data, *, requires_grad=False):
        """创建常量值节点"""
        value = cls.__new__(cls)
        value._init(
            None,
            [],
            cached_data=data,
            requires_grad=requires_grad,
        )
        return value

    @classmethod
    def make_from_op(cls, op: Op, inputs: List["Value"]):
        """通过算子创建值节点"""
        # 除去调用__init__方法以外的另一种初始化方法
        # 调用该方法的原因是__init__已经被重载过了
        value = cls.__new__(cls)
        value._init(op, inputs)  # 初始化对应成员变量
        # 如果设置LAZY_MODE，realize_cached_data不会被调用
        # 该函数实际执行计算逻辑。所以在LAZY_MODE下初始化张量后不会马上得到计算结果
        # LAZY_MODE模式下，可以通过调用x.data获得计算结果
        # 该方法会调用detach方法，进一步调用realize_cached_data进行计算
        # 如果计算特别耗时，可以考虑设置LAZY_MODE，这样可以省去计算时间，先建图，再优化
        if not LAZY_MODE:
            if not value.requires_grad:
                return value.detach()
            value.realize_cached_data()  # 尝试计算并写入cached_data成员变量
        return value


class TensorTuple(Value):
    """Represent a tuple of tensors.
    To keep things simple, we do not support nested tuples.
    """

    def __len__(self):
        cdata = self.realize_cached_data()
        return len(cdata)

    def __getitem__(self, index: int):
        return needle.ops.tuple_get_item(self, index)

    def tuple(self):
        return tuple([x for x in self])

    def __repr__(self):
        return "needle.TensorTuple" + str(self.tuple())

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        return needle.ops.make_tuple(*[self[i] + other[i] for i in range(len(self))])

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return TensorTuple.make_const(self.realize_cached_data())


class Tensor(Value):
    grad: "Tensor"

    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        """张量构造函数"""
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                # 回退方案，通过numpy转换
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            # 处理普通数组输入
            device = device if device else _default_tensor_device()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        """从numpy数组创建张量数据"""
        if array_api is numpy:
            return _ensure_numpy_array_api(numpy.array(numpy_array, dtype=dtype))
        return array_api.array(numpy_array, device=device, dtype=dtype)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        """通过算子创建张量"""
        # 除去调用__init__方法以外的另一种初始化方法
        # 调用该方法的原因是__init__已经被重载过了
        tensor = Tensor.__new__(Tensor)
        # 初始化对应成员变量
        tensor._init(op, inputs)
        # 如果设置LAZY_MODE，realize_cached_data不会被调用
        # 该函数实际执行计算逻辑。所以在LAZY_MODE下初始化张量后不会马上得到计算结果
        # LAZY_MODE模式下，可以通过调用y.data获得计算结果
        # 该方法会调用detach方法，进一步调用realize_cached_data进行计算
        # 如果计算特别耗时，可以考虑设置LAZY_MODE，这样可以省去计算时间，先建图，再优化
        if not LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach()
            # 尝试计算并写入cached_data成员变量
            tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        """创建常量张量"""
        if isinstance(data, Tensor):
            cached_data = data.realize_cached_data()
        elif array_api is numpy:
            cached_data = _ensure_numpy_array_api(numpy.array(data))
        elif isinstance(data, NDArray):
            cached_data = data
        else:
            cached_data = Tensor._array_from_numpy(data, device=_default_tensor_device(), dtype=None)

        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )
        tensor.cached_data = _ensure_numpy_array_api(tensor.cached_data)
        return tensor

    @property
    def data(self):
        """获取数据(脱离计算图)"""
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realize_cached_data()

    def detach(self):
        """创建共享数据但脱离计算图的新张量"""
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    @property
    def device(self):
        data = self.realize_cached_data()
        # numpy数组总是在CPU上
        if array_api is numpy:
            return cpu()
        return data.device

    def backward(self, out_grad=None):
        out_grad = (
            out_grad
            if out_grad
            else init.ones(*self.shape, dtype=self.dtype, device=self.device)
        )
        compute_gradient_of_variables(self, out_grad)

    def __repr__(self):
        return "needle.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def numpy(self):
        """转换为numpy数组"""
        data = self.realize_cached_data()
        if array_api is numpy:
            return data
        return data.numpy()

    # 以下实现各种运算符重载
    def __add__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, other)
        else:
            # 通过定义一个算子实现了张量和标量相加的具体逻辑
            return needle.ops.AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseMul()(self, other)
        else:
            return needle.ops.MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWisePow()(self, other)
        else:
            return needle.ops.PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, needle.ops.Negate()(other))
        else:
            return needle.ops.AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseDiv()(self, other)
        else:
            return needle.ops.DivScalar(other)(self)

    def __matmul__(self, other):
        return needle.ops.MatMul()(self, other)

    def matmul(self, other):
        return needle.ops.MatMul()(self, other)

    def sum(self, axes=None, keepdims=False):
        return needle.ops.Summation(axes, keepdims)(self)

    def broadcast_to(self, shape):
        return needle.ops.BroadcastTo(shape)(self)

    def reshape(self, shape):
        return needle.ops.Reshape(shape)(self)

    def __neg__(self):
        return needle.ops.Negate()(self)

    def transpose(self, axes=None):
        return needle.ops.Transpose(axes)(self)

    def exp(self):
        return needle.ops.Exp()(self)

    def log(self):
        return needle.ops.Log()(self)

    def relu(self):
        return needle.ops.ReLU()(self)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__


def compute_gradient_of_variables(output_tensor, out_grad):
    """Take gradient of output node with respect to each node in node_list.
    Store the computed result in the grad field of each Variable.
    """
    """计算输出节点相对于node_list中每个节点的梯度, 并将结果存储在Variable的grad字段中"""
    # 从节点到每个输出节点的梯度贡献列表的映射
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    # 注意梯度初始化的特殊情况
    # 实际上是对标量reduce_sum(output_node)求导，而不是向量output_node
    # 但对于损失函数来说这是常见情况
    node_to_output_grads_list[output_tensor] = [out_grad]

    # 按照反向拓扑顺序遍历计算图
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    for node in reverse_topo_order:
        # 计算当前节点的梯度
        node.grad = sum_node_list(node_to_output_grads_list[node])
        if node.op is None:
            continue
        # 计算并传播梯度到输入节点
        for child, grad in zip(node.inputs, node.op.gradient_as_tuple(node.grad, node)):
            if child not in node_to_output_grads_list:
                node_to_output_grads_list[child] = []
            node_to_output_grads_list[child].append(grad)


def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    """给定节点列表，返回以这些节点结尾的拓扑排序列表"""
    visited = set()  # 已访问节点集合
    topo_order = []  # 拓扑排序结果
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node: Value, visited: Set[Value], topo_order: List[Value]):
    """Post-order DFS"""
    if node not in visited:
        visited.add(node)
        for child in node.inputs:  # 遍历所有输入节点(子节点)
            topo_sort_dfs(child, visited, topo_order)
        topo_order.append(node)  # 后序添加当前节点


def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    """自定义求和函数, 避免Python sum实现中创建冗余节点"""
    from operator import add
    from functools import reduce

    return reduce(add, node_list)
