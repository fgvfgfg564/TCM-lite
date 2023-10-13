import matplotlib.pyplot as plt
import os
import numpy as np
import json

folder = os.path.split(__file__)[0]
fdata = os.path.join(folder, 'data.json')

with open(fdata, 'r') as f:
    data = json.load(f)

score = data['gen_score']
psnr = data['gen_psnr']
time = data['gen_time']
n_gen = len(score)

plt.plot(np.arange(n_gen), score)
plt.title("Gen. score")
plt.xlabel("Gen")
plt.ylabel("score")
plt.grid()
plt.savefig(os.path.join(folder, "score.png"))
plt.clf()

plt.plot(np.arange(n_gen), psnr)
plt.title("Gen. psnr")
plt.xlabel("Gen")
plt.ylabel("psnr")
plt.grid()
plt.savefig(os.path.join(folder, "psnr.png"))
plt.clf()

plt.plot(np.arange(n_gen), time)
plt.title("Gen. time")
plt.xlabel("Gen")
plt.ylabel("time")
plt.grid()
plt.savefig(os.path.join(folder, "time.png"))