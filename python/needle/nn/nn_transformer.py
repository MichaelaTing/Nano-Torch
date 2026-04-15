import math
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_sequence import Embedding
from .nn_basic import Parameter, Module, ReLU, Dropout, LayerNorm1d, Linear, Sequential


class MultiHeadAttention(Module):
    """
    The multi-head self attention module.
    """

    def __init__(
        self,
        *,
        dropout=0.0,
        causal=False,
        device=None,
        dtype="float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.causal = causal
        self.dropout = Dropout(dropout)

    def create_causal_mask(self, i, j, device):
        """
        return a triangular causal mask.
        """
        # 未来位点填充极小值，使 softmax 后概率约为 0
        mask = -np.finfo(np.float32).max * np.triu(
            np.ones((1, 1, i, j), dtype=np.float32), j - i + 1
        )
        return Tensor(mask, device=device, dtype=self.dtype, requires_grad=False)

    def matmul(self, a, b):
        """
        batched matrix multiplication;
        """
        a_shape = (*a.shape[:-1], 1, *a.shape[-1:])
        a = a.reshape(a_shape)

        b_shape = (*b.shape[:-2], 1, *b.shape[-2:])
        b = b.reshape(b_shape)

        broadcast_shape = list(a_shape)
        broadcast_shape[-2] = b_shape[-2]
        a = a.broadcast_to(broadcast_shape)

        broadcast_shape = list(b_shape)
        broadcast_shape[-3] = a_shape[-3]
        b = b.broadcast_to(broadcast_shape)

        return (a * b).sum(len(a.shape) - 1)

    def softmax(self, logit):
        """
        The softmax function;
        """
        # 先减去每行最大值，提升 softmax 数值稳定性
        max_val = Tensor(
            logit.realize_cached_data().max(axis=3),
            device=logit.device,
            dtype=logit.dtype,
            requires_grad=False,
        )

        max_val = max_val.reshape((*logit.shape[:-1], 1))
        max_val = max_val.broadcast_to(logit.shape)

        probs = ops.exp(logit - max_val)

        denom = probs.sum(axes=3)
        denom = denom.reshape((*logit.shape[:-1], 1))
        denom = denom.broadcast_to(logit.shape)

        return probs / denom

    def forward(
        self,
        q,
        k,
        v,
    ):
        """
        The forward function of the MultiHeadAttention activation function.
        Input: three states q, k, v, with shape (batch_size, num_head, seq_len, dim_head)
        Output: the activation output `result` and attention softmax probability `probs` (with dropout applied)
        """
        batch_size, num_head, q_len, q_dim = q.shape
        _, _, k_len, k_dim = k.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim

        result = None
        probs = None

        # 注意力分数: QK^T / sqrt(d_k)
        probs = self.matmul(q, k) / (q_dim**0.5)
        if self.causal:
            mask = self.create_causal_mask(q_len, k_len, device=q.device)
            mask = mask.broadcast_to((batch_size, num_head, q_len, k_len))
            probs = probs + mask
        probs = self.dropout(self.softmax(probs))
        result = self.matmul(probs, ops.transpose(v))

        return result, probs


class AttentionLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        out_features: int = None,
        dropout=0.0,
        causal=True,
        device=None,
        dtype="float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        if k_features is None:
            k_features = q_features
        if v_features is None:
            v_features = q_features
        if out_features is None:
            out_features = q_features

        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

        self.num_head = num_head
        self.dim_head = dim_head

        self.prenorm_q = LayerNorm1d(q_features, device=device, dtype=dtype)
        self.prenorm_k = LayerNorm1d(k_features, device=device, dtype=dtype)
        self.prenorm_v = LayerNorm1d(v_features, device=device, dtype=dtype)

        inner_dim = num_head * dim_head

        self.q_projection = Linear(
            q_features, inner_dim, bias=False, device=device, dtype=dtype
        )
        self.k_projection = Linear(
            k_features, inner_dim, bias=False, device=device, dtype=dtype
        )
        self.v_projection = Linear(
            v_features, inner_dim, bias=False, device=device, dtype=dtype
        )

        self.attn = MultiHeadAttention(
            dropout=dropout, causal=causal, device=device, dtype=dtype
        )

        self.out_projection = Linear(
            inner_dim, out_features, bias=False, device=device, dtype=dtype
        )

    def forward(
        self,
        q,
        k=None,
        v=None,
    ):
        """
        The forward function of the self-attention layer.
        Input: `q` with shape (batch_size, q_len, q_dim)
               `k` (if not None) with shape (batch_size, kv_len, k_dim)
               `v` (if not None) with shape (batch_size, kv_len, v_dim)
        Output: the output `result` with shape (batch_size, kv_len, out_features)
        """

        if k is None:
            k = q
        if v is None:
            v = q

        batch_size, q_len, q_dim = q.shape
        _, k_len, k_dim = k.shape
        _, _, v_dim = v.shape

        result = None

        q_norm = self.prenorm_q(q.reshape((batch_size * q_len, q_dim)))
        k_norm = self.prenorm_k(k.reshape((batch_size * k_len, q_dim)))
        v_norm = self.prenorm_v(v.reshape((batch_size * k_len, v_dim)))

        q_w = self.q_projection(q_norm)  # 查询向量投影后形状为 (bs * q_len, d * head)
        k_w = self.k_projection(k_norm)  # 键向量投影后形状为 (bs * k_len, d * head)
        v_w = self.v_projection(v_norm)  # 值向量投影后形状为 (bs * k_len, d * head)

        q_w = q_w.reshape((batch_size, q_len, self.num_head, self.dim_head))
        k_w = k_w.reshape((batch_size, k_len, self.num_head, self.dim_head))
        v_w = v_w.reshape((batch_size, k_len, self.num_head, self.dim_head))

        q_w = q_w.transpose(axes=(1, 2))  # 调整为多头注意力所需布局
        k_w = k_w.transpose(axes=(1, 2))  # 调整为多头注意力所需布局
        v_w = v_w.transpose(axes=(1, 2))  # 调整为多头注意力所需布局

        out, probs = self.attn(q_w, k_w, v_w)  # 经过注意力聚合后的多头输出
        out = out.transpose(axes=(1, 2)).reshape(
            (batch_size * q_len, self.num_head * self.dim_head)
        )  # 将多头结果重新拼回特征维

        result = self.out_projection(out).reshape(
            (batch_size, q_len, self.out_features)
        )  # 投影回最终输出维度

        return result


class TransformerLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout=0.0,
        causal=True,
        device=None,
        dtype="float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.dropout = Dropout(dropout)
        self.attn_layer = AttentionLayer(
            q_features,
            num_head,
            dim_head,
            dropout=dropout,
            causal=causal,
            device=device,
            dtype=dtype,
        )

        self.norm = LayerNorm1d(q_features, device=device, dtype=dtype)
        self.linear1 = Linear(q_features, hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(hidden_size, q_features, device=device, dtype=dtype)
        self.relu = ReLU()

    def forward(self, x):
        """
        The forward function of a Transformer Layer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """

        batch_size, seq_len, x_dim = x.shape

        # 残差分支 1: Self-Attention
        x = x + self.dropout(self.attn_layer(x))

        # 残差分支 2: FFN
        y = self.norm(x.reshape((batch_size * seq_len, x_dim)))
        y = self.dropout(self.relu(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        x = x + y.reshape((batch_size, seq_len, x_dim))

        return x


class Transformer(Module):

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        *,
        num_head: int = 8,
        dim_head: int = 32,
        dropout=0.0,
        causal=True,
        device=None,
        dtype="float32",
        batch_first=False,
        sequence_len=2048,
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first

        self.pos_embed = Embedding(
            sequence_len, embedding_size, device=device, dtype=dtype
        )
        self.transformer_layers = Sequential(
            *[
                TransformerLayer(
                    embedding_size,
                    num_head=num_head,
                    dim_head=dim_head,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    causal=causal,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

    def create_timestamp_tensor(self, batch_size: int, seq_len: int):
        # 位置索引张量: [0, 1, ..., seq_len-1]
        ts = np.arange(seq_len).reshape(seq_len, 1)
        ts = Tensor(ts, device=self.device, dtype=self.dtype, requires_grad=False)
        return ts.broadcast_to((seq_len, batch_size))

    def forward(self, x, h=None):

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        batch_size, seq_len, input_dim = x.shape
        ts = self.create_timestamp_tensor(batch_size, seq_len)  # 生成每个位置的时间戳索引
        pos_embedding = self.pos_embed(ts)  # 查表得到位置嵌入
        pos_embedding = ops.transpose(
            pos_embedding, axes=(0, 1)
        )  # 调整成与输入相同的 batch-first 布局

        x = x + pos_embedding  # 将位置编码加到输入特征上
        x = self.transformer_layers(x)  # 依次通过多层 Transformer

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)
