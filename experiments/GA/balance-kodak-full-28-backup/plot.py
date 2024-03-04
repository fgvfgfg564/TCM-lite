import matplotlib.pyplot as plt
import json
import os
from matplotlib.ticker import MultipleLocator
from matplotlib import rcParams

plt.rcParams.update({'font.size': 14})
plt.rcParams['font.family'] = 'DejaVu Serif'
basedir = os.path.split(__file__)[0]

# bpp=0.9937

EXTRA_DATA = {
    # "JPEG": (31.65, 0.018),
    # "WebP": (33.79, 0.024),
    # "BPG": (35.36, 0.385),
    "VTM": (37.8622, 0.4246),
    # "cheng2020": (36.48, 11.957),
}

avg_bpp = 0.8026

figsize = (6, 4)
fig, ax = plt.subplots(figsize=figsize)

def jsonload(filename):
    with open(filename, "r") as f:
        return json.load(f)

ours = jsonload(os.path.join(basedir, "mixed/results.json"))

T = []
D = []

for k, v in ours.items():
    T.append(v['avg_t_dec'])
    D.append(v['avg_psnr'])

ax.plot(T[:-2], D[:-2], label="Ours", marker='o', color='blue')

points = ["EVC_LL", "EVC_LM", "EVC_LS", "TCM"]

for point in points:
    data = jsonload(os.path.join(basedir, point, "results.json"))
    t_dec = data['avg_t_dec']
    psnr = data['avg_psnr']
    ax.scatter(t_dec, psnr, label=point, marker='*', s=75)

for point, (psnr, t_dec) in EXTRA_DATA.items():
    ax.scatter(t_dec, psnr, label=point, marker='^')

# ax.set_xscale('log')
ax.set_xlabel("Decode time/s")
ax.set_ylabel("PSNR/dB")
ax.legend(loc='lower right')

ax.set_axisbelow(True)
ax.xaxis.set_minor_locator(MultipleLocator(0.04))
ax.yaxis.set_minor_locator(MultipleLocator(0.02))
ax.grid(linestyle='-', alpha=0.5, linewidth=0.8, which='major')
ax.grid(linestyle='-', alpha=0.5, linewidth=0.3, which='minor')

# plt.title("Kodak")

plt.tight_layout()
plt.savefig(os.path.join(basedir, "kodak-balance-qp28.png"), dpi=300)
plt.savefig(os.path.join(basedir, "kodak-balance-qp28.pdf"))