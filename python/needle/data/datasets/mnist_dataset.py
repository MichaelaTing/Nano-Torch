from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        def read(file):
            with gzip.GzipFile(file) as f:
                # 按 IDX 文件头解析 dtype 与维度信息
                _, _, dtype, ndim = struct.unpack("4b", f.read(4))
                shape = struct.unpack(f">{ndim}i", f.read(4 * ndim))
                return np.frombuffer(f.read(), dtype=dtype_map[dtype]).reshape(shape)

        dtype_map = {
            8: np.uint8,
            9: np.int8,
            11: np.int16,
            12: np.int32,
            13: np.float32,
            14: np.float64,
        }
        self.X = read(image_filename)
        self.y = read(label_filename)
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        X = self.X[index]
        y = self.y[index]
        if isinstance(index, slice):
            # 批量访问时逐样本执行 transform，再统一归一化到 [0,1]
            X = np.stack([self.apply_transforms(x) for x in X])
            X = X.reshape(len(X), -1) / np.float32(255.0)
        else:
            X = self.apply_transforms(X)
            X = X.reshape(-1) / np.float32(255.0)
        return X, y

    def __len__(self) -> int:
        return len(self.y)
