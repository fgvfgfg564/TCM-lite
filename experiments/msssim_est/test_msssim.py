from pytorch_msssim import ms_ssim
import torch

A = torch.rand([1, 1, 512, 512]) / 4.0
B = torch.rand([1, 1, 512, 512])

A[:, :, 252:260, :] = 0

print("A-B", ms_ssim(A, B, 1))

A_half_1 = A[:, :, :256, :]
A_half_2 = A[:, :, 256:, :]
B_half_1 = B[:, :, :256, :]
B_half_2 = B[:, :, 256:, :]


print("half", (ms_ssim(A_half_1, B_half_1, 1.0) + ms_ssim(A_half_2, B_half_2, 1.0)) / 2)

A_half_half = torch.cat([A_half_1, torch.flip(A_half_1, dims=[2])], dim=2)
B_half_half = torch.cat([B_half_1, torch.flip(B_half_1, dims=[2])], dim=2)

print(ms_ssim(A_half_1, B_half_1, 1.0), ms_ssim(A_half_2, B_half_2, 1))

print(
    "half-half",
    (ms_ssim(A_half_half, B_half_half, 1) + ms_ssim(A_half_half, B_half_half, 1)) / 2,
)


# A_zero = torch.cat([A[:, :, :256, :], torch.zeros([1, 1, 256, 512])], dim=2)
# B_zero = torch.cat([B[:, :, :256, :], torch.zeros([1, 1, 256, 512])], dim=2)

# print("half+zero", ms_ssim(A_zero, B_zero, 1))
