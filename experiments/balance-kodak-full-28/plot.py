import matplotlib.pyplot as plt
import json
import os
from matplotlib.ticker import MultipleLocator
from matplotlib import rcParams
import math

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

EXTRA_DATA_MSSSIM = {
    "VTM": (0.99004207, 0.4246),
}

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

# MS-SSIM results

def to_msssim_psnr(msssim):
    if isinstance(msssim, float):
        return -10*math.log10(1. - msssim)
    results = []
    for m in msssim:
        results.append(-10*math.log10(1. - m))
    return results

fig, ax = plt.subplots(figsize=figsize)

T = []
D = []

ours = jsonload(os.path.join(basedir, "mixed/results.json"))
for k, v in ours.items():
    T.append(v['avg_t_dec'])
    D.append(to_msssim_psnr(v['avg_ms_ssim']))

ax.plot(T[:-2], D[:-2], label="Ours", marker='o', color='blue')

points = ["EVC_LL", "EVC_LM", "EVC_LS", "TCM"]

for point in points:
    data = jsonload(os.path.join(basedir, point, "results.json"))
    t_dec = data['avg_t_dec']
    psnr = to_msssim_psnr(data['avg_ms_ssim'])
    ax.scatter(t_dec, psnr, label=point, marker='*', s=75)

for point, (psnr, t_dec) in EXTRA_DATA_MSSSIM.items():
    ax.scatter(t_dec, to_msssim_psnr(psnr), label=point, marker='^')

# ax.set_xscale('log')
ax.set_xlabel("Decode time/s")
ax.set_ylabel("MS-SSIM [dB]")
ax.legend(loc='lower right')

ax.set_axisbelow(True)
ax.xaxis.set_minor_locator(MultipleLocator(0.04))
ax.yaxis.set_minor_locator(MultipleLocator(0.02))
ax.grid(linestyle='-', alpha=0.5, linewidth=0.8, which='major')
ax.grid(linestyle='-', alpha=0.5, linewidth=0.3, which='minor')

# plt.title("Kodak")

plt.tight_layout()
plt.savefig(os.path.join(basedir, "kodak-balance-qp28-ms-ssim.png"), dpi=300)
plt.savefig(os.path.join(basedir, "kodak-balance-qp28-ms-ssim.pdf"))