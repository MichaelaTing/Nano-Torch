import os

import numpy as np
from needle import backend_ndarray as nd
from needle import Tensor


class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.words = set()

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        if word not in self.word2idx:
            # 词表按首次出现顺序分配连续 id
            uid = len(self.idx2word)
            self.word2idx[word] = uid
            self.idx2word.append(word)
        return self.word2idx[word]

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        return len(self.idx2word)


class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """

    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(self._resolve_split_path(base_dir, "train"), max_lines)
        self.test = self.tokenize(self._resolve_split_path(base_dir, "test"), max_lines)

    def _resolve_split_path(self, base_dir, split):
        candidates = [
            os.path.join(base_dir, f"ptb.{split}.txt"),
            os.path.join(base_dir, f"{split}.txt"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Cannot find PTB {split} file in {base_dir}")

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ids = []
        eos_id = self.dictionary.add_word("<eos>")

        def tokenize_one_line(line):
            words = line.split()
            for word in words:
                ids.append(self.dictionary.add_word(word))
            # 每行末尾补 <eos>，显式标记句子结束
            ids.append(eos_id)

        with open(path, "r") as f:
            if max_lines:
                for _ in range(max_lines):
                    line = f.readline()
                    if line == "":
                        break
                    tokenize_one_line(line)
            else:
                for line in f:
                    tokenize_one_line(line)
        return ids


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    # 丢弃不能整除 batch_size 的尾部，保证后续 reshape 合法
    nbatch = len(data) // batch_size
    data = np.array(data[: nbatch * batch_size], dtype=np.float32).reshape((nbatch, batch_size))
    return data


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    tot_seqlen = batches.shape[0]
    assert i < tot_seqlen - 1
    if i + bptt + 1 > tot_seqlen:
        # 最后一个片段长度可能小于 bptt
        X = batches[i:-1, :]
        y = batches[i + 1 :, :].flatten()
    else:
        X = batches[i : i + bptt, :]
        y = batches[i + 1 : i + 1 + bptt, :].flatten()
    if dtype is None:
        dtype = "float32"
    return Tensor(X, device=device, dtype=dtype), Tensor(y, device=device, dtype=dtype)
