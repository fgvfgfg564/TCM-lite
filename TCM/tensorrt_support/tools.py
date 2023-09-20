import torch
import torch_tensorrt
import os

from .registry import VIRTUALIZE_TENSORRT_MODULES
from .modules import TorchTensorRTPlaceholder

def compile(model: torch.nn.Module, output_folder):
    new_state_dict = {}
    for name, child in model.named_children():
        print(f"Compiling module: {name}")
        if child.tensorrt_compilable:
            input_shape = child.input_shape
            example_inputs = torch.randn(input_shape)
            scripted_model = torch.jit.script(child, example_inputs=example_inputs)
            trt_model = torch_tensorrt.ts.compile(scripted_model, 
                inputs= [torch_tensorrt.Input(input_shape, dtype=torch.half)],
                enabled_precisions= {torch.float, torch.half},
                debug=True,
                require_full_compilation=True,
            )
            torch.jit.save(trt_model, os.path.join(output_folder, name+".ts"))
        else:
            child_state_dict = child.state_dict(prefix=name+".")
            new_state_dict.update(child_state_dict)
    torch.save(new_state_dict, os.path.join(output_folder, "state_dict.pth.tar"))

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
            model._modules[name] = torch.jit.load(trt_filename)