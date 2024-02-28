import os
import json
import matplotlib.pyplot as plt

BASEDIR = os.path.split(__file__)[0]
datadir = os.path.join(BASEDIR, "results/1000/100")

results = {"2.0": []}

for bpg_qp in [30, 31, 32, 33]:
    for w_time in [0.029, 0.080, 0.200, 0.800]:
        for bytes_sigma in [2., 8., 32., 128., 256., 512., 1024., 2048., 4096.]:
            try:
                datajson = os.path.join(datadir, str(bpg_qp), str(w_time), "0.01/False/0.2", str(bytes_sigma), "DOG_4507_statistics.json")
                with open(datajson, "r") as f:
                    data = json.load(f)
                
                gen_score = data["gen_score"][-1]
                results.setdefault(str(bytes_sigma), [])
                results[str(bytes_sigma)].append(gen_score)
            except Exception as e:
                pass

all_data = []
labels = []
    
for k, v in results.items():
    all_data.append(v)
    labels.append(k)

plt.boxplot(all_data, vert=True, patch_artist=True, labels=labels, showfliers=False)
plt.xlabel("$\sigma_b$")
plt.ylabel("score")
plt.savefig(os.path.join(BASEDIR, "ablation-bytes-sigma.png"), dpi=300)
plt.savefig(os.path.join(BASEDIR, "ablation-bytes-sigma.pdf"))