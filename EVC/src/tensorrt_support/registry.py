from typing import Any
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

class OneTimeInputShapeRecorderHook:
    def __init__(self, module: torch.nn.Module) -> None:
        self.module = module
        self.handle = module.register_forward_hook(self)

    def __call__(self, module, inputs, outputs) -> Any:
        module.input_shape = inputs[0].shape
        print(f"Recorded input shape: {module.input_shape}")
        self.handle.remove()
    
    @classmethod
    def attach(cls, module):
        cls(module)

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
            OneTimeInputShapeRecorderHook.attach(self)
    
    cls.__init__ = new_init_
    return cls

def maybe_tensorrt(module):
    """
    Add to the initialization of a module. To replace the module if tensorrt is used.
    """
    if VIRTUALIZE_TENSORRT_MODULES:
        return TorchTensorRTPlaceholder()
    else:
        module.tensorrt_compilable = True
        OneTimeInputShapeRecorderHook.attach(module)
        return module