import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim),
            )
        ),
        nn.ReLU(),
    )


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[
            ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob)
            for _ in range(num_blocks)
        ],
        nn.Linear(hidden_dim, num_classes),
    )


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    if opt is not None:
        model.train()
        opt.reset_grad()
    else:
        model.eval()
    softmax_loss = nn.SoftmaxLoss()
    total_loss, total_error, len_dataset = 0, 0, len(dataloader.dataset)
    for x, y in dataloader:
        logit = model(x)
        loss = softmax_loss(logit, y)
        loss.backward()
        if opt is not None:
            opt.step()
            opt.reset_grad()
        total_loss += loss.numpy().item() * y.shape[0]
        total_error += (logit.numpy().argmax(-1) != y.numpy()).sum().item()
    return total_error / len_dataset, total_loss / len_dataset


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    train_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/train-images-idx3-ubyte.gz",
        f"{data_dir}/train-labels-idx1-ubyte.gz",
    )
    train_dataloader = ndl.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        train_err, train_loss = epoch(train_dataloader, model, opt)
    test_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/t10k-images-idx3-ubyte.gz", f"{data_dir}/t10k-labels-idx1-ubyte.gz"
    )
    test_dataloader = ndl.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    test_err, test_loss = epoch(test_dataloader, model, None)
    return train_err, train_loss, test_err, test_loss
