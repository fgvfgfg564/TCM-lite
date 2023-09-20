import torch
import torch_tensorrt

VIRTUALIZE_TENSORRT_MODULES = False

def tensorrt_mark(module, input_shape):
    module.use_tensorrt = True
    module.input_shape = input_shape

    return module

def tensorrt_compiled_class(cls):
    