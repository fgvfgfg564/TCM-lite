import torch
import torch_tensorrt
from .modules import TorchTensorRTPlaceholder

VIRTUALIZE_TENSORRT_MODULES = False

def tensorrt_mark(module, input_shape):
    module.use_tensorrt = True
    module.input_shape = input_shape

    return module

def tensorrt_compiled_module(cls):
    """
    Any class wrapped by this decorator must have 'input_shape' property
    """
    if VIRTUALIZE_TENSORRT_MODULES:
        return TorchTensorRTPlaceholder
    else:
        setattr(cls, 'tensorrt_compilable', True)
        return cls