from bin.math import normalize_to_target
import numpy as np

X = np.array([0, 0, 0, 9])
X_max = np.array([3,5,7,9])

for target in range(30):
    print(normalize_to_target(X, X_max, target))