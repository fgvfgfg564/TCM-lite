import os
import numpy as np
import matplotlib.pyplot as plt


folder = os.path.split(__file__)[0]

plt.rcParams.update({'font.size': 14})
plt.rcParams['font.family'] = 'DejaVu Serif'
basedir = os.path.split(__file__)[0]

mse1 = 5e-4
mse2 = 1e-3
t1 = 0.1
t2 = 0.02
N = 32

psnrs = []
times = []
scores = []

for n in range(N+1):
    m = N - n
    mse = (n*mse1 + m*mse2) / N
    psnr = -10*np.log10(mse)

    t = n*t1 + m*t2
    score = t - psnr
    psnrs.append(psnr)
    times.append(t)
    scores.append(score)

# plt.plot(range(N+1), psnrs, label='PSNR')
# plt.grid()
# plt.xlabel('$n$')
# plt.ylabel("PSNR/dB")
# plt.savefig(os.path.join(folder, "n-PSNR.png"), dpi=300)
# plt.savefig(os.path.join(folder, "n-PSNR.pdf"))
# plt.close()

# plt.plot(range(N+1), times, label='time')
# plt.grid()
# plt.xlabel('$n$')
# plt.ylabel("time/s")
# plt.savefig(os.path.join(folder, "n-time.png"), dpi=300)
# plt.savefig(os.path.join(folder, "n-time.pdf"))
# plt.close()

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(range(N+1), scores, label='score', zorder=1)

ax.scatter(0, scores[0], marker='^', color='blue', zorder=2)
ax.text(0.8, scores[0]-0.015, 'Local Optimum')
ax.scatter(N, scores[-1], marker='*', color='red', zorder=2)
ax.text(N-13, scores[-1], 'Global Optimum')

ax.grid()
ax.set_xlabel('$n_A$')
ax.set_ylabel("$\mathcal{L}_{rdc}=\\alpha t_{dec}-PSNR$")
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(folder, "n-score.png"), dpi=300)
plt.savefig(os.path.join(folder, "n-score.pdf"))