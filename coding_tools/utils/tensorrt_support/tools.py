import torch
import torch_tensorrt
import os

from .modules import TorchTensorRTPlaceholder

def compile(model: torch.nn.Module, output_folder):
    try:
        os.makedirs(output_folder)
    except FileExistsError:
        pass
    model.eval()
    new_state_dict = {}
    for name, child in model.named_children():
        if hasattr(child, "tensorrt_compilable") and child.tensorrt_compilable and hasattr(child, "input_shape"):
            input_shape = child.input_shape
            print(f"Compiling module: {name} into TensorRT TorchScript; input_shape={input_shape}")
            with torch_tensorrt.logging.debug():
                trt_model = torch_tensorrt.compile(child, 
                    inputs= [torch_tensorrt.Input(input_shape, dtype=torch.half)],
                    enabled_precisions={torch.half},
                )
            torch.jit.save(trt_model, f"g_s_6.ts")
            torch.jit.save(trt_model, os.path.join(output_folder, name+".ts"))
        else:
            child_state_dict = child.state_dict(prefix=name+".")
            new_state_dict.update(child_state_dict)
    torch.save(new_state_dict, os.path.join(output_folder, "state_dict.pth.tar"))

class TypeConversionWrapper(torch.nn.Module):
    def __init__(self, mod):
        super().__init__()
        self.mod = mod
    
    def forward(self, x):
        y = self.mod(x)
        y = y.half()
        return y

def load_weights(model: torch.nn.Module, state_dict_folder):
    """
    Load model weights from a folder which originates from compiling script; The tensorrt modules will be 
    directly loaded and other modules load weights.
    """

    state_dict = torch.load(os.path.join(state_dict_folder, "state_dict.pth.tar"))
    model.load_state_dict(state_dict, strict=False)

    for name, child in model.named_children():
        if isinstance(child, TorchTensorRTPlaceholder):
            trt_filename = os.path.join(state_dict_folder, name+".ts")
            trt_module = torch.jit.load(trt_filename)
            trt_module = TypeConversionWrapper(trt_module)
            model._modules[name] = trt_module