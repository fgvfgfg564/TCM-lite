import numpy as np

def is_sorted(arr):
    return np.array_equal(arr, np.sort(arr))

class LinearInterpolation:
    def __init__(self, X, Y):
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        self.check_input()

    def check_input(self):
        if len(self.X) != len(self.Y):
            raise ValueError("X and Y must have the same length.")
        if not is_sorted(self.X):
            raise ValueError(f"X must be sorted. Found: {self.X}")
    
    def __call__(self, x):
        return self.interpolate(x)

    def interpolate(self, x):
        idx = np.searchsorted(self.X, x)
        idx = np.clip(idx, 1, len(self.X) - 1)  # Ensure idx is within bounds

        x0, x1 = self.X[idx - 1], self.X[idx]
        y0, y1 = self.Y[idx - 1], self.Y[idx]

        slope = (y1 - y0) / (x1 - x0)
        interpolated_y = y0 + slope * (x - x0)
        return interpolated_y

    def derivative(self, x):
        idx = np.searchsorted(self.X, x)
        idx = np.clip(idx, 1, len(self.X) - 1)  # Ensure idx is within bounds

        x0, x1 = self.X[idx - 1], self.X[idx]
        y0, y1 = self.Y[idx - 1], self.Y[idx]

        slope = (y1 - y0) / (x1 - x0)
        return slope

def safe_softmax(x: np.ndarray):
    x = x - np.max(x)
    x = np.maximum(x, -16)
    y = np.exp(x)
    return y / np.sum(y)

# class LinearRegression:
#     def __init__(self, X, Y):
#         self.X = np.asarray(X)
#         self.Y = np.asarray(Y)
#         self.check_input()

#     def check_input(self):
#         if len(self.X) != len(self.Y):
#             raise ValueError("X and Y must have the same length.")

#     def fit(self):
#         # Compute linear regression parameters (slope and intercept)
#         self.slope, self.intercept = np.polyfit(self.X, self.Y, 1)

#     def interpolate(self, x):
#         # Linear interpolation function
#         return self.slope * x + self.intercept

#     def derivative(self, x):
#         # Derivative of the linear regression function (constant slope)
#         return self.slope

def waterfill(X, k):
    if np.sum(X) <= k:
        return X
    
    N = len(X)
    sorted_indexes = np.argsort(X)
    sorted_X = X[sorted_indexes]
    pref_sorted_X = np.cumsum(sorted_X)
    results = np.zeros_like(X)

    # Search the end of fully occupied cell
    l = -1
    r = N - 1
    while(l < r - 1):
        mid = (l+r) // 2
        if mid == -1:
            y = 0
            t = 0
        else:
            y = pref_sorted_X[mid]
            t = sorted_X[mid]
        y += t * (N - 1 - mid)

        if y <= k:
            l = mid
        else:
            r = mid
    
    results[sorted_indexes[:l+1]] = X[sorted_indexes[:l+1]]
    d = (k - pref_sorted_X[l]) / (N - 1 - l)
    results[sorted_indexes[l+1:]] = d

    return results

def normalize_to_target(X: np.ndarray, X_max, target, inplace=False):
    if not inplace:
        X = X.copy()
    remain = target - np.sum(X)
    if remain > 0:
        inverse = False
        space = X_max - X
    else:
        inverse = True
        space = X
        remain = -remain
    
    fill = waterfill(space, remain)
    if inverse:
        X -= fill
    else:
        X += fill

    assert(np.all(X >= 0))
    assert(np.all(X <= X_max))
    assert(np.sum(X) <= target)
    
    return X