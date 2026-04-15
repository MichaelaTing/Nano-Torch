import numpy as np
from ..autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from ..backend_selection import cpu


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """

    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(order)
        # 按 batch_size 将样本索引切成若干小批次
        self.ordering = np.array_split(
            order, range(self.batch_size, len(self.dataset), self.batch_size)
        )
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == len(self.ordering):
            raise StopIteration
        # 先从 Dataset 取样，再按字段聚合并堆叠成 Tensor 批次
        data = [self.dataset[index] for index in self.ordering[self.idx]]
        self.idx += 1
        return tuple(
            Tensor(np.stack(x), device=cpu(), requires_grad=False) for x in zip(*data)
        )
