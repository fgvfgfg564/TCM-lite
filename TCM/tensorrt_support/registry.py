import torch
import torch_tensorrt
from .modules import TorchTensorRTPlaceholder

VIRTUALIZE_TENSORRT_MODULES = False

def tensorrt_mark(module, input_shape):
    module.use_tensorrt = True
    module.input_shape = input_shape

    return module

class InitTRTModelWithPlaceholder:
    def __enter__(self):
        global VIRTUALIZE_TENSORRT_MODULES
        VIRTUALIZE_TENSORRT_MODULES = True
    
    def __exit__(self, *args, **kwargs):
        global VIRTUALIZE_TENSORRT_MODULES
        VIRTUALIZE_TENSORRT_MODULES = False

def tensorrt_compiled_module(cls):
    """
    Any class wrapped by this decorator must have 'input_shape' property
    """

    old_init_ = cls.__init__

    def new_init_(self, *args, **kwargs):
        print(VIRTUALIZE_TENSORRT_MODULES)
        if VIRTUALIZE_TENSORRT_MODULES:
            self.__class__ = TorchTensorRTPlaceholder
            TorchTensorRTPlaceholder.__init__(self, *args, **kwargs)
        else:
            old_init_(self, *args, **kwargs)
            setattr(self, 'tensorrt_compilable', True)
    
    cls.__init__ = new_init_
    return cls