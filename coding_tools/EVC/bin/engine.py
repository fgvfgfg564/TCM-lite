import os
import torch

from ...coding_tool import LICToolBase
from ...register import register_tool

from ..src.models import build_model, image_model
from ..src.utils.stream_helper import get_state_dict


@register_tool("EVC")
class ModelEngine(LICToolBase):
    MODELS = {
        "EVC_LS": "EVC_LS_MD.pth.tar",
        "EVC_LS_large": "EVC_LS_large.pth.tar",
        "EVC_LS_mid": "EVC_LS_mid.pth.tar",
        "EVC_LM": "EVC_LM_MD.pth.tar",
        "EVC_LL": "EVC_LL.pth.tar",
        "EVC_LL_large": "EVC_LL_large.pth.tar",
        # "EVC_ML": 'EVC_ML_MD.pth.tar',
        # "EVC_SL": 'EVC_SL_MD.pth.tar',
        # "EVC_MM": 'EVC_MM_MD.pth.tar',
        # "EVC_SS": 'EVC_SS_MD.pth.tar',
        # "Scale_EVC_SL": 'Scale_EVC_SL_MDRRL.pth.tar',
        # "Scale_EVC_SS": 'Scale_EVC_SS_MDRRL.pth.tar',
    }
    MIN_CTU_SIZE = 256

    def __init__(self, model_name, dtype, ctu_size) -> None:
        super().__init__(model_name, dtype, ctu_size)

        # load model
        model_path = self.get_model_path(model_name)

        i_state_dict = get_state_dict(model_path)
        i_frame_net = build_model(model_name[:6], ec_thread=True)
        i_frame_net.load_state_dict(i_state_dict, verbose=False)
        i_frame_net = i_frame_net.cuda()
        i_frame_net.eval()

        i_frame_net.update(force=True)
        self.i_frame_net: image_model.EVC = i_frame_net.to(dtype)
        self.q_scales = i_state_dict["q_scale"].to(dtype)
        self.q_scale_min = self.q_scales[-1]
        self.q_scale_max = self.q_scales[0]
        self.cuda()

    def lic_compress(self, img_block, q_scale):
        self.eval()
        with torch.no_grad():
            q_scale = self._q_scale_mapping(q_scale).to(img_block.device)
            bit_stream = self.i_frame_net.compress(img_block, q_scale)["bit_stream"]
            return bit_stream

    def lic_decompress(self, bit_stream, h, w, q_scale):
        self.eval()
        with torch.no_grad():
            q_scale = self._q_scale_mapping(q_scale).cuda()
            recon_img = self.i_frame_net.decompress(bit_stream, h, w, q_scale)["x_hat"]
            return recon_img

    @classmethod
    def get_model_path(cls, model_name):
        model_path = cls.MODELS[model_name]
        file_folder = os.path.split(__file__)[0]
        model_path = os.path.join(file_folder, "../checkpoints", model_path)
        return model_path
