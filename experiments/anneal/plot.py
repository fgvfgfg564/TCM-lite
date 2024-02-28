import matplotlib.pyplot as plt
import numpy as np
import os

N = 1000
x = np.arange(N+1)
T = 0.75*(1-(x/N))**2+0.25

ROOTDIR = os.path.split(__file__)[0]

plt.plot(x, T)
plt.grid()
plt.xlabel("Gen.")
plt.ylabel("$T$")
plt.savefig(os.path.join(ROOTDIR, "anneal.png"), dpi=300)
plt.savefig(os.path.join(ROOTDIR, "anneal.pdf"))