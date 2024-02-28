import os
import json

from tools.plotters import *

folder = os.path.split(__file__)[0]


def load_single(json_filename):
    with open(json_filename, "r") as f:
        data = json.load(f)
    return data["avg_psnr"], data["avg_t_dec"]


data_LL_fname = os.path.join(folder, "../EVC-LL-only-bpg32/results.json")
data_LM_fname = os.path.join(folder, "../EVC-LM-only-bpg32/results.json")
data_LS_fname = os.path.join(folder, "../EVC-LS-only-bpg32/results.json")
data_all_fname = os.path.join(folder, "results.json")

LL_psnr, LL_t_dec = load_single(data_LL_fname)
LM_psnr, LM_t_dec = load_single(data_LM_fname)
LS_psnr, LS_t_dec = load_single(data_LS_fname)
all_psnr, all_t_dec = load_single(data_all_fname)

Ds = [[LL_psnr], [LM_psnr], [LS_psnr], [all_psnr]]
Ts = [[LL_t_dec], [LM_t_dec], [LS_t_dec], [all_t_dec]]
labels = ["EVC_LL", "EVC_LM", "EVC_LS", "Mixed"]

plot_0_0(
    Ds,
    Ts,
    labels,
    "ooos",
    os.path.join(folder, "b_t"),
    "1857.11-Class A",
    "t_dec",
    "PSNR",
)
