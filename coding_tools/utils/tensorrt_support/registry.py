from typing import Any
import torch
import torch_tensorrt
from .modules import TorchTensorRTPlaceholder

VIRTUALIZE_TENSORRT_MODULES = False

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
        print(f"Recorded input shape: {module.input_shape}", id(module))
        self.handle.remove()
    
    @classmethod
    def attach(cls, module):
        cls(module)

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