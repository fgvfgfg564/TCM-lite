import os
import json
import numpy as np

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})
plt.rcParams['font.family'] = 'DejaVu Serif'
BASEDIR = os.path.split(__file__)[0]
datadir = os.path.join(BASEDIR, "results/1000/100")

# for n_disturb in range(1, 10):
#     label = f"$n_d={n_disturb}$"
#     data_filename = os.path.join(BASEDIR, str(n_disturb), "results/1000/100/30/0.0/0.05/False/0.2/256/DOG_4507_statistics.json")
#     with open(data_filename, "r") as f:
#         data = json.load(f)
    
#     score_curve = data["gen_score"]
#     gen = np.arange(len(score_curve))

#     plt.plot(gen, score_curve, label=label)

# plt.xlabel("Gen")
# plt.ylabel("Score")
# plt.grid()
# plt.legend()
# plt.savefig(os.path.join(BASEDIR, "ablation-anti-disturb.png"), dpi=300)
# plt.savefig(os.path.join(BASEDIR, "ablation-anti-disturb.pdf"))
# plt.close()

for n_disturb in range(1, 10):
    label = f"$n_d={n_disturb}$"
    data_filename = os.path.join(BASEDIR, str(n_disturb), "results/1000/1000/30/0.0/0.05/False/0.2/256/DOG_4507_statistics.json")
    with open(data_filename, "r") as f:
        data = json.load(f)
    
    score_curve = data["gen_psnr"]
    gen = np.arange(len(score_curve))

    plt.plot(gen, score_curve, label=label)

PSNR_EVC = 38.441
PSNR_TCM = 39.296

plt.axhline(y=PSNR_EVC, color='grey', linestyle='--')
plt.axhline(y=PSNR_TCM, color='red', linestyle='--')

plt.text(500, PSNR_TCM+0.02, "Theoretical Optimum", color='red')
plt.text(500, PSNR_EVC+0.02, "Interference Method", color='grey')

plt.xlabel("Gen")
plt.ylabel("PSNR/dB")
plt.xlim(left=0.)
plt.ylim((PSNR_EVC - 0.02, PSNR_TCM+0.08))
plt.grid()
plt.legend(loc='center right')
plt.savefig(os.path.join(BASEDIR, "ablation-anti-disturb-psnr.png"), dpi=300)
plt.savefig(os.path.join(BASEDIR, "ablation-anti-disturb-psnr.pdf"))