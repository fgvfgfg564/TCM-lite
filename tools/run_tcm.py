from TCM.models.tcm import TCM_vbr
import torch

if __name__ == "__main__":
    model = TCM_vbr(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=128, M=320)
    model = model.cuda()
    
    input_img = torch.zeros((1, 3, 512, 512), device='cuda')
    input_lmbda = torch.tensor([0.], device='cuda')

    model(input_img, input_lmbda)