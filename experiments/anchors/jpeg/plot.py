import os
import json

from math import sqrt, pow
from tools.plotters import *
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.interpolate import griddata

plt.rcParams.update({"font.size": 14})
plt.rcParams["font.family"] = "DejaVu Serif"
basedir = os.path.split(__file__)[0]

folder = os.path.split(__file__)[0]
figsize = (6, 4)


def load_single(json_filename):
    with open(json_filename, "r") as f:
        data = json.load(f)
    return data


data_filename = os.path.join(folder, "results.json")
data = load_single(data_filename)

# Draw R-D

fig, ax = plt.subplots(figsize=figsize)

for quality, dt_data in data.items():
    bpp = dt_data["avg_bpp"]
    ax.plot(
        dt_data["avg_t_dec"],
        dt_data["avg_psnr"],
        label=f"bpp={bpp:.3f}",
        marker=".",
        linewidth=0.7,
    )

ax.legend()
ax.grid()
ax.set_xlabel("Dec. Time / s")
ax.set_ylabel("PSNR / dB")
# ax.set_title("DOG_4507.png")
# ax.set_xlim(0.)
plt.tight_layout()
plt.savefig(os.path.join(folder, "d-t.png"), dpi=300)
plt.savefig(os.path.join(folder, "d-t.pdf"))
plt.close()
