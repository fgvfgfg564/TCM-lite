import os
import json
import matplotlib.pyplot as plt

BASEDIR = os.path.split(__file__)[0]
datadir = os.path.join(BASEDIR, "results/1000/100")

results = {"1e-09": []}

for bpg_qp in [30, 31, 32, 33]:
    for w_time in [0.029, 0.080, 0.200, 0.800]:
        for boltzmann_k in [1e-9, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 1]:
            try:
                datajson = os.path.join(datadir, str(bpg_qp), str(w_time), str(boltzmann_k), "False/0.2/512/DOG_4507_statistics.json")
                with open(datajson, "r") as f:
                    data = json.load(f)
                
                gen_score = data["gen_score"][-1]
                results.setdefault(str(boltzmann_k), [])
                results[str(boltzmann_k)].append(gen_score)
            except Exception as e:
                pass

all_data = []
labels = []
    
for k, v in results.items():
    all_data.append(v)
    labels.append(k)

plt.boxplot(all_data, vert=True, patch_artist=True, labels=labels, showfliers=False)
plt.xlabel("$k$")
plt.ylabel("score")
plt.savefig(os.path.join(BASEDIR, "ablation-boltzmann-k.png"), dpi=300)
plt.savefig(os.path.join(BASEDIR, "ablation-boltzmann-k.pdf"))