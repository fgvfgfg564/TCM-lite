import numpy as np
from scipy.interpolate import PchipInterpolator
from typing_extensions import Callable, Literal, Union


def is_sorted(arr):
    return np.array_equal(arr, np.sort(arr))


class PolyValWrapper(np.ndarray):
    def __new__(cls, input_array):
        # Input array is an array-like object that you want to turn into a MyArray
        obj = np.asarray(input_array).view(cls)
        # Add your extra information or initialization here
        return obj

    def __call__(self, x):
        return np.polyval(self, x)

    def derivative(self):
        return PolyValWrapper(np.polyder(self))

    def posroot(self):
        if len(self) > 3:
            raise ValueError("Only supports quadratic funcs")

        a = self[0]
        b = self[1]
        c = self[2]

        o = b**2 - 4 * a * c
        if o < 0:
            return np.inf
        return (-b + np.sqrt(o)) / (2 * a)


def fit3d(X, Y) -> PolyValWrapper:
    fit = np.polyfit(X, Y, 3)
    fit = PolyValWrapper(fit)
    return fit


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


class Warpped4DFitter:
    METHOD_DICT = {"fit3d": fit3d, "pchip": PchipInterpolator}

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        method: Union[Literal["fit3d"], Literal["pchip"]],
    ) -> None:
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        self.method = str(method)
        self.curve: Union[PchipInterpolator, PolyValWrapper] = self.METHOD_DICT[
            self.method
        ](
            X,
            Y,
        )
        self.X_min = self.X.min()
        self.X_max = self.X.max()
        self.d = self.curve.derivative()

    def __call__(self, x):
        return self.interpolate(x)

    def interpolate(self, x):
        return self.curve(x)

    def derivative(self, x: float) -> float:
        return self.d(x)

    def dump(self, filename):
        np.savez_compressed(filename, X=self.X, Y=self.Y, method=self.method)

    @classmethod
    def load(cls, filename):
        loaded = np.load(filename)
        return cls(X=loaded["X"], Y=loaded["Y"], method=loaded["method"])


def safe_softmax(x: np.ndarray):
    x = x - np.max(x)
    x = np.maximum(x, -16)
    y = np.exp(x)
    return y / np.sum(y)


def safe_SA_prob(delta, T):
    if delta / T < 60:
        p = 1.0 / (1.0 + np.exp(delta / T))
    else:
        p = 0
    return p


def binary_search(
    f: Callable[[float], float],
    target: float,
    x_min: float,
    x_max: float,
    epsilon: float,
    debug: bool = False,
) -> float:
    """
    Binary search on an ascending function
    """
    l = x_min
    r = x_max
    while l < r - epsilon:
        mid = (l + r) / 2
        u = f(mid)
        if u <= target:
            l = mid
        else:
            r = mid
        if debug:
            print(f"l={l}; r={r}")
    return l


def waterfill(X, k):  # O(N)
    if k <= 0:
        return np.zeros_like(X)

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
    while l < r - 1:
        mid = (l + r) // 2
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

    results[sorted_indexes[: l + 1]] = X[sorted_indexes[: l + 1]]
    if l == -1:
        pref = 0
    else:
        pref = pref_sorted_X[l]
    d = (k - pref) / (N - 1 - l)
    results[sorted_indexes[l + 1 :]] = d

    return results


def normalize_to_target(X: np.ndarray, X_min, X_max, target):
    X = np.maximum(X, X_min)
    X = np.minimum(X, X_max)

    X = X - X_min
    X_max = X_max - X_min
    target = target - sum(X_min)

    remain = target - np.sum(X)
    if remain > 0:
        inverse = False
        space = X_max - X
    else:
        inverse = True
        space = X.copy()
        remain = -remain

    fill = waterfill(space, remain)

    if inverse:
        X -= fill
    else:
        X += fill

    return X + X_min
