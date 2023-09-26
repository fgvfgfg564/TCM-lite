import torch
import torch_tensorrt
import os

from .modules import TorchTensorRTPlaceholder
from utils import Timer

def compile(model: torch.nn.Module, output_folder):
    model.eval()
    new_state_dict = {}
    for name, child in model.named_children():
        if hasattr(child, "tensorrt_compilable") and child.tensorrt_compilable:
            input_shape = child.input_shape
            print(f"Compiling module: {name} into TensorRT TorchScript; input_shape={input_shape}")
            print(child)
            example_inputs = torch.randn(input_shape)
            scripted_model = torch.jit.script(child, example_inputs=[example_inputs])
            print(scripted_model.code)
            with torch_tensorrt.logging.info():
                trt_model = torch_tensorrt.ts.compile(scripted_model, 
                    inputs= [torch_tensorrt.Input(input_shape, dtype=torch.half)],
                    enabled_precisions= {torch.float, torch.half},
                    debug=True,
                    require_full_compilation=True,
                )
            torch.jit.save(trt_model, f"g_s_6.ts")
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
            with Timer("Load tensorrt script"):
                trt_filename = os.path.join(state_dict_folder, name+".ts")
                model._modules[name] = torch.jit.load(trt_filename)