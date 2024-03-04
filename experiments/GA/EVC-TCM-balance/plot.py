import os
import json

from math import sqrt, pow
from tools.plotters import *
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.interpolate import griddata

plt.rcParams.update({'font.size': 14})
plt.rcParams['font.family'] = 'DejaVu Serif'
basedir = os.path.split(__file__)[0]

color0, color1, color2, color3, colorp = ('#211A3E', '#453370', '#A597B6', '#FEF3E8', '#F3D266')

positions = [0., 0.33, 0.66, 1.]
cm = LinearSegmentedColormap.from_list(name="KeQing", colors=list(zip(positions, [color0, color1, color2, color3])), N=256)

folder = os.path.split(__file__)[0]
figsize = (6, 4)

def load_single(json_filename):
    with open(json_filename, "r") as f:
        data = json.load(f)
    return data


data_filename = os.path.join(folder, "results.json")
data = load_single(data_filename)

# Draw D-T

fig, ax = plt.subplots(figsize=figsize)

for bpg_qp, wdt_data in data.items():
    D = []
    T = []
    for w_time, dt_data in wdt_data.items():
        D.append(dt_data['avg_psnr'])
        T.append(dt_data['avg_t_dec'])
        bpp = dt_data['avg_bpp']
    ax.plot(T, D, label=f'bpp={bpp:.3f}', marker='.', linewidth=0.7)

ax.legend()
ax.grid()
ax.set_xlabel("Dec. Time / s")
ax.set_ylabel("PSNR / dB")
# ax.set_title("DOG_4507.png")
ax.set_xlim(0.)
plt.tight_layout()
plt.savefig(os.path.join(folder, "d-t.png"), dpi=300)
plt.savefig(os.path.join(folder, "d-t.pdf"))
plt.close()



# Draw R-D

fig, ax = plt.subplots(figsize=figsize)

D = {}
R = {}

for bpg_qp, wdt_data in data.items():
    for w_time, dt_data in wdt_data.items():
        D.setdefault(w_time, [])
        R.setdefault(w_time, [])
        D[w_time].append(dt_data['avg_psnr'])
        R[w_time].append(dt_data['avg_bpp'])

for w_time in D.keys():
    ax.plot(R[w_time], D[w_time], label=f'$\\alpha={w_time[7:]}$', marker='.', linewidth=0.7)

ax.legend(fontsize=10)
ax.grid()
ax.set_xlabel("Bpp")
ax.set_ylabel("PSNR / dB")
# ax.set_title("DOG_4507.png")
plt.tight_layout()
plt.savefig(os.path.join(folder, "r-d.png"), dpi=300)
plt.savefig(os.path.join(folder, "r-d.pdf"))
plt.close()

# Draw 3D
plt.rcParams.update({'font.size': 11})
fig = plt.figure(figsize=figsize)

# Add a 3-D subplot
ax = fig.add_subplot(111, projection='3d')

Ds = []
Ts = []
Bs = []

for bpg_qp, wdt_data in data.items():
    D = []
    T = []
    B = []
    for w_time, dt_data in wdt_data.items():
        D.append(dt_data['avg_psnr'])
        T.append(dt_data['avg_t_dec'])
        B.append(dt_data['avg_bpp'])
    Ds.append(D)
    Ts.append(T)
    Bs.append(B)

def interpolate_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    x1 = x.flatten()
    y1 = y.flatten()
    z1 = z.flatten()
    data = np.array([x1, y1]).T

    new_x = np.linspace(x.min(), x.max(), 100)
    new_y = np.linspace(y.min(), y.max(), 100)
    grid_x, grid_y = np.meshgrid(new_x, new_y)

    plane = griddata(data, z1, (grid_x, grid_y), method='linear')
    return grid_x, grid_y, plane

Bs = np.array(Bs)
Ts = np.array(Ts)
Ds = np.array(Ds)

Bi, Ti, Di = interpolate_plane(Bs, Ts, Ds)

ax.view_init(30, -25, 0)

ax.plot_surface(Bi, Ti, Di, linewidth=2.0, cmap=cm, shade=True)
# ax.plot_surface(Bs, Ts, Ds)
ax.scatter(Bs, Ts, Ds, marker='.', c=colorp)
ax.set_xlim(ax.get_xlim()[::-1])

# ax.scatter(ax.get_xlim()[0], Ts, Ds, marker='.')
# ax.scatter(Bs, ax.get_ylim()[1], Ds, marker='.')
# ax.scatter(Bs, Ts, ax.get_zlim()[0], marker='.')

# ax.w_xaxis.pane.set_edgecolor("black")
# ax.w_zaxis.pane.set_edgecolor("black")
# ax.w_yaxis.pane.set_edgecolor("black")

ax.set_xlabel('Bit-rate [bpp]')
ax.set_ylabel('Dec.Time / s')
ax.set_zlabel('PSNR / dB')
# ax.set_title('DOG_4507.png')
plt.tight_layout()
plt.savefig(os.path.join(folder, "3d.png"), dpi=300)
plt.savefig(os.path.join(folder, "3d.pdf"))
plt.close()