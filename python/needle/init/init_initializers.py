import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, shape=None, gain=1.0, **kwargs):
    # Xavier 均匀初始化：尽量保持前后层激活方差稳定
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    if shape is None:
        return rand(fan_in, fan_out, low=-a, high=a, **kwargs)
    else:
        return rand(*shape, low=-a, high=a, **kwargs)


def xavier_normal(fan_in, fan_out, shape=None, gain=1.0, **kwargs):
    a = gain * math.sqrt(2 / (fan_in + fan_out))
    if shape is None:
        return randn(fan_in, fan_out, std=a, **kwargs)
    else:
        return randn(*shape, std=a, **kwargs)


def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    # Kaiming 均匀初始化：针对 ReLU 按 fan_in 缩放
    a = math.sqrt(6 / fan_in)
    if shape is None:
        return rand(fan_in, fan_out, low=-a, high=a, **kwargs)
    else:
        return rand(*shape, low=-a, high=a, **kwargs)


def kaiming_normal(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    a = math.sqrt(2 / fan_in)
    if shape is None:
        return randn(fan_in, fan_out, std=a, **kwargs)
    else:
        return randn(*shape, std=a, **kwargs)
