
from torch import nn
from collections import OrderedDict

class ClassInstantier(OrderedDict):
    """
    Allows to dynamically instantiate classes from a dictionary of classes.
    """
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)

ACTIVATIONS_FUNCTIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "prelu": nn.PReLU,
    "rrelu": nn.RReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "softplus": nn.Softplus,
    "softsign": nn.Softsign,
    "hardshrink": nn.Hardshrink,
    "tanhshrink": nn.Tanhshrink,
    "softshrink": nn.Softshrink,
    "hardtanh": nn.Hardtanh,
    "log_sigmoid": nn.LogSigmoid,
    "hardswish": nn.Hardswish,
}

ACTIVATION = ClassInstantier(ACTIVATIONS_FUNCTIONS)

def get_activation(activation_string):
    if activation_string in ACTIVATION:
        return ACTIVATION[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACTIVATIONS_FUNCTIONS mapping {list(ACTIVATION.keys())}")