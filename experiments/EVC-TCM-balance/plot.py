import os
import json

from tools.plotters import *

folder = os.path.split(__file__)[0]


def load_single(json_filename):
    with open(json_filename, "r") as f:
        data = json.load(f)
    return data


data_filename = os.path.join(folder, "results.json")
data = load_single(data_filename)

D = []
T = []
label = "EVC-LL + TCM"

for w, w_data in data.items():
    D.append(w_data["avg_psnr"])
    T.append(w_data["avg_t_dec"])

plot_0_0(
    [D],
    [T],
    [label],
    "o",
    os.path.join(folder, "b_t"),
    "DOG_4507.png",
    "t_dec/s",
    "PSNR/dB",
)
