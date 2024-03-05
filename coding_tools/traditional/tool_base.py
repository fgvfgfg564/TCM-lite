import torch
from ..coding_tool import CodingToolBase


def PSNR(x, y):
    mse = torch.mean((x - y) ** 2)
    return -10 * torch.log10(mse)


class TraditionalCodingToolBase(CodingToolBase):
    PLATFORM = 'numpy'

    def compress_block(self, img_block: torch.Tensor, q_scale: float) -> bytes:
        """
        输入:
        - img_block 图像块，格式为torch.Tensor，值范围为[0, 1]，形状为[1, 3, H, W]
        - q_scale 控制目标码率，对应QP，值范围为[0, 1]
            q_scale=1对应编码工具有效范围内的最低码率，q_scale=0对应最高码率
        返回：
        - 编码出的比特流，格式为bytes
        """
        raise NotImplemented

    def decompress_block(
        self, bit_stream: bytes, h: int, w: int, q_scale: float
    ) -> torch.Tensor:
        """
        输入：
        - bit_stream 即self.compress_block函数的输出，比特流
        - h, w 输出图像的大小，若不需要可忽略
        - q_scale 同self.compress_block的q_scale；q_scale和bit_stream应当与编码时一致；不需要可忽略
        返回：
        - 解码图像，格式应当转换为值域[0, 1]，格式为torch.float32的torch.Tensor，形状为[1, 3, H, W]，与compress_block一致
        """
        raise NotImplemented

    def self_test(self, test_block: torch.Tensor, q_scale: float) -> None:
        _, c, h, w = test_block.shape()
        bits = self.compress_block(test_block, q_scale)
        recon = self.decompress_block(bits, h, w, q_scale)

        bpp = len(bits) * 8 / h / w
        psnr = PSNR(test_block, recon)
        print(f"bpp={bpp:.6f}; psnr={psnr:.6f}")
