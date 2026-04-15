from ..data_basic import Dataset


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        # 同一索引下并行返回多路数组字段（如特征与标签）
        return tuple([a[i] for a in self.arrays])
