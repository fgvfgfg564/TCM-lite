from __future__ import annotations

from typing_extensions import Dict, TypeVar, Self

import abc
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from typing_extensions import Callable, Literal, Union


def is_sorted(arr):
    return np.array_equal(arr, np.sort(arr))


T = TypeVar("T", np.ndarray, float)


class DerivableFunc(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x: T) -> T:
        pass

    @abc.abstractmethod
    def derivative(self) -> DerivableFunc:
        pass


class PolyValWrapper(np.ndarray, DerivableFunc):
    def __new__(cls, input_array):
        # Input array is an array-like object that you want to turn into a MyArray
        obj = np.asarray(input_array).view(cls)
        # Add your extra information or initialization here
        return obj

    def __call__(self, x):
        return np.polyval(self, x)

    def derivative(self):
        return PolyValWrapper(np.polyder(self))


@dataclass
class ExpKModel(DerivableFunc):
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray = np.asarray(0.0)
    SCALE: float = field(default=1000, init=False, repr=False)

    def __call__(self, x: T) -> T:
        B, X = np.meshgrid(self.b, x)
        result = np.power(1 + np.exp(-B), -X / self.SCALE) @ self.a + self.c
        if not isinstance(x, np.ndarray):
            result = result.item()
        return result

    def derivative(self) -> ExpKModel:
        a_new = -self.a * np.log(1 + np.exp(-self.b)) / self.SCALE
        b_new = self.b
        return ExpKModel(a_new, b_new, np.asarray(0.0))


class Fitter(abc.ABC):
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> None:
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        self.curve = self.fit()
        self.X_min = self.X.min()
        self.X_max = self.X.max()
        self.d = self.curve.derivative()

    def R2(self, curve):
        y_mean = self.Y.mean()
        sum1 = 0
        sum2 = 0
        for x, y in zip(self.X, self.Y):
            y_pred = curve(x)
            sum1 += (y - y_pred) ** 2
            sum2 += (y - y_mean) ** 2
        return 1 - sum1 / sum2

    def mse(self, curve):
        sqe = 0
        for x, y in zip(self.X, self.Y):
            y_pred = curve(x)
            sqe += (y - y_pred) ** 2
        return sqe / len(self.X)

    def __call__(self, x):
        return self.interpolate(x)

    def interpolate(self, x):
        return self.curve(x)

    def derivative(self, x: float) -> float:
        return self.d(x)

    def dump(self, filename):
        np.savez_compressed(filename, X=self.X, Y=self.Y)

    @classmethod
    def load(cls, filename):
        loaded = np.load(filename)
        return cls(X=loaded["X"], Y=loaded["Y"])

    @abc.abstractmethod
    def fit(self) -> DerivableFunc:
        pass


class FitCubic(Fitter):
    def fit(self) -> PolyValWrapper:
        fit = np.polyfit(self.X, self.Y, 3)
        fit = PolyValWrapper(fit)
        return fit


class FitKExp(Fitter):

    def __init__(self, X: np.ndarray, Y: np.ndarray, K=3) -> None:
        self.K = K
        super().__init__(X, Y)
        print("MSE=", self.mse(self.curve))
        r2 = self.R2(self.curve)
        print("R2=", r2)

    def fit(self):
        a_init = np.random.rand(self.K) * 2 * self.Y[0] / self.K
        b_init = np.zeros([self.K]) - 0.05
        init_value = np.concatenate([a_init, b_init], axis=0)
        curve = ExpKModel(a_init.copy(), b_init.copy())

        y_std2 = ((self.Y - self.Y.mean()) ** 2).sum()

        def objective_func(ab: np.ndarray):
            # R2 loss
            a = ab[: self.K]
            b = ab[self.K :]
            curve.a = a
            curve.b = b
            objective = 1.0 - self.R2(curve)
            return objective

        def objective_gradient(ab: np.ndarray):
            a = ab[: self.K]
            b = ab[self.K :]
            curve.a = a
            curve.b = b
            y_pred = curve(self.X)
            d_r2_y_pred = 2.0 / y_std2 * (y_pred - self.Y)
            da = [
                (d_r2_y_pred * np.power(1 + np.exp(-b[i]), -self.X / curve.SCALE)).sum()
                for i in range(self.K)
            ]
            db = [
                (
                    d_r2_y_pred
                    * a[i]
                    * (-self.X / curve.SCALE)
                    * np.power(1 + np.exp(-b[i]), -(self.X / curve.SCALE + 1))
                    * (-np.exp(-b[i]))
                ).sum()
                for i in range(self.K)
            ]
            da = np.asarray(da)
            db = np.asarray(db)
            # print(a, b, da, db, flush=True)
            return np.concatenate([da, db], axis=0)

        bounds = [(1e-9, None)] * self.K + [(-2, 10)] * self.K

        result = minimize(
            objective_func,
            init_value,
            jac=objective_gradient,
            method="SLSQP",
            bounds=bounds,
            options={
                "ftol": 1e-12,
                "maxiter": 10000,
            },
            tol=1e-12,
        )

        ans = result.x
        return ExpKModel(ans[: self.K], ans[self.K :])
        # when x=0, f(x)=sum(a) <= 1.

    def dump(self, filename):
        np.savez_compressed(
            filename, X=self.X, Y=self.Y, K=self.K, a=self.curve.a, b=self.curve.b
        )

    @classmethod
    def load(cls, filename) -> Self:
        loaded = np.load(filename)
        obj = cls.__new__(cls)
        obj.X = loaded["X"]
        obj.Y = loaded["Y"]
        obj.K = loaded["K"]
        obj.curve = ExpKModel(a=loaded["a"], b=loaded["b"], c=np.asarray(0))
        obj.X_min = obj.X.min()
        obj.X_max = obj.X.max()
        obj.d = obj.curve.derivative()
        return obj


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
    func: Callable[[float], float],
    target: float,
    x_min: float,
    x_max: float,
    epsilon: float,
    f_epsilon: float = None,
    debug: bool = False,
) -> float:
    """
    Binary search on an ascending function
    """
    l = x_min
    r = x_max
    while l < r - epsilon:
        mid = (l + r) / 2
        u = func(mid)
        if u <= target:
            if f_epsilon is not None and u >= target - f_epsilon:
                return mid
            l = mid
            fl = u
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
