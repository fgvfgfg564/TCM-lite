from __future__ import annotations
import random
from typing import cast

from torch import NoneType
from typing_extensions import (
    Dict,
    TypeVar,
    Self,
    Callable,
    Literal,
    Union,
    Optional,
    override,
)

import abc
from dataclasses import field

import numpy as np
from .type import NDArray
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize


def is_sorted(arr):
    return np.array_equal(arr, np.sort(arr))


class DerivableFunc(abc.ABC):
    def __init__(self, c: Optional[NDArray] = None) -> None:
        if c is None:
            c = np.asarray(0.0)
        self.c = c

    def __call__(self, x: NDArray) -> NDArray:
        return self.funccall(x) + self.c

    @abc.abstractmethod
    def funccall(self, x: NDArray) -> NDArray:
        pass

    @abc.abstractmethod
    def derivative(self) -> DerivableFunc:
        pass

    @abc.abstractmethod
    def dump(self) -> Dict[str, NDArray]:
        pass

    @classmethod
    def load(cls, cfg: Dict[str, NDArray]):
        return cls(**cfg)


# class PolyValWrapper(np.ndarray, DerivableFunc):
#     def __new__(cls, input_array):
#         # Input array is an array-like object that you want to turn into a MyArray
#         obj = np.asarray(input_array).view(cls)
#         # Add your extra information or initialization here
#         return obj

#     def funccall(self, x):
#         return np.polyval(self, x)

#     def derivative(self):
#         return PolyValWrapper(np.polyder(self))


class ExpKModel(DerivableFunc):
    SCALE = 1000

    def __init__(self, c: Optional[NDArray] = None, *, a, b) -> None:
        super().__init__(c)
        self.a = a
        self.b = b

    def __repr__(self) -> str:
        return f"<y = {self.a} * ({1 + np.exp(-self.b)}) ** (-x/{self.SCALE}) + {self.c}; a={self.a}; b={self.b}; c={self.c}>"

    @override
    def funccall(self, x: NDArray) -> NDArray:
        B, X = np.meshgrid(self.b, x)
        result = np.power(1 + np.exp(-B), -X / self.SCALE) @ self.a
        if not isinstance(x, np.ndarray):
            result = result.item()
        result = cast(NDArray, result)
        return result

    @override
    def derivative(self) -> ExpKModel:
        a_new = -self.a * np.log(1 + np.exp(-self.b)) / self.SCALE
        b_new = self.b.copy()
        return ExpKModel(a=a_new, b=b_new)

    @override
    def dump(self) -> Dict[str, NDArray]:
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
        }


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

    def maxerror(self, curve):
        errs = []
        for x, y in zip(self.X, self.Y):
            y_pred = curve(x)
            errs.append(np.abs(y - y_pred))
        return np.max(errs)

    def ms_rel_err(self, curve):
        y_pred = curve(self.X)
        return np.mean((y_pred / self.Y - 1.0) ** 2)

    def __call__(self, x):
        return self.interpolate(x)

    def interpolate(self, x):
        return self.curve(x)

    def derivative(self, x: NDArray) -> NDArray:
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


# class FitCubic(Fitter):
#     def fit(self) -> PolyValWrapper:
#         fit = np.polyfit(self.X, self.Y, 3)
#         fit = PolyValWrapper(fit)
#         return fit


class FitKExp(Fitter):

    def __init__(self, X: np.ndarray, Y: np.ndarray, K=3, retry=True) -> None:
        self.K = K
        super().__init__(X, Y)

        while True:
            ms_rel_err = self.ms_rel_err(self.curve)

            if retry and ms_rel_err > 1e-4 and len(self.X) > 6:
                print(
                    f"Warning: bad fit. ms_rel_err={ms_rel_err:.6f}; curve={self.curve}"
                )
                print("Bad fit, retrying... Ignore the first & last element")
            else:
                break
            self.X = self.X[1:-1]
            self.Y = self.Y[1:-1]
            self.curve = self.fit()
            self.X_min = self.X.min()
            self.X_max = self.X.max()
            self.d = self.curve.derivative()

    def fit(self):
        midi = len(self.Y) // 2

        lgy = np.log(self.Y)
        lg_fit = np.polyfit(self.X, lgy, 1)

        a_init = np.random.rand(self.K)
        a_init /= a_init.sum()
        a_init *= np.exp(lg_fit[1])

        t = lg_fit[0]
        if t >= -np.log(np.exp(-10) + 1) / ExpKModel.SCALE:
            b_init = np.zeros([self.K]) - 10
        else:
            b_init = np.zeros([self.K]) - np.log(np.exp(-t * ExpKModel.SCALE) - 1)

        init_value = np.concatenate([a_init, b_init], axis=0)

        def objective_func(ab: np.ndarray):
            # R2 loss
            a = ab[: self.K]
            b = ab[self.K :]
            curve = ExpKModel(a=a.copy(), b=b.copy(order="A"))
            Y_pred = curve(self.X)
            loss = (Y_pred / self.Y - 1.0) ** 2
            loss = np.mean(loss)
            return loss

        def objective_gradient(ab: np.ndarray):
            a = ab[: self.K]
            b = ab[self.K :]
            curve = ExpKModel(a=a.copy(), b=b.copy(order="A"))
            y_pred = curve(self.X)
            d_loss_y_pred = 2.0 * (y_pred / self.Y - 1) / self.Y
            da = [
                (
                    d_loss_y_pred * np.power(1 + np.exp(-b[i]), -self.X / curve.SCALE)
                ).mean()
                for i in range(self.K)
            ]
            db = [
                (
                    d_loss_y_pred
                    * a[i]
                    * (-self.X / curve.SCALE)
                    * np.power(1 + np.exp(-b[i]), -(self.X / curve.SCALE + 1))
                    * (-np.exp(-b[i]))
                ).mean()
                for i in range(self.K)
            ]
            da = np.asarray(da)
            db = np.asarray(db)
            # print(a, b, da, db, flush=True)
            return np.concatenate([da, db], axis=0)

        bounds = [(1e-3, 2**16)] * self.K + [(-5, None)] * self.K

        result = minimize(
            objective_func,
            init_value,
            jac=objective_gradient,
            method="SLSQP",
            bounds=bounds,
            options={
                "ftol": 1e-12,
                "maxiter": 1000000,
            },
            tol=1e-12,
        )

        ans = result.x
        ans = cast(NDArray, ans)
        check = np.all(ans[: self.K] >= 0)
        if not check:
            print(self.X, self.Y, ans)
            raise ValueError("Fit failed! Some A < 0")
        return ExpKModel(a=ans[: self.K].copy(), b=ans[self.K :].copy(order="A"))
        # when x=0, f(x)=sum(a) <= 1.

    def dump(self, filename):
        curve_cfg = self.curve.dump()
        curve_cfg = {"curve." + k: v for k, v in curve_cfg.items()}
        np.savez_compressed(filename, X=self.X, Y=self.Y, K=self.K, **curve_cfg)

    @classmethod
    def load(cls, filename) -> Self:
        loaded: Dict[str, NDArray] = np.load(filename)
        obj = cls.__new__(cls)
        obj.X = loaded["X"]
        obj.Y = loaded["Y"]
        obj.K = loaded["K"]
        obj.X_min = obj.X.min()
        obj.X_max = obj.X.max()
        curve_cfg = {
            k.split(sep=".", maxsplit=2)[1]: v
            for k, v in loaded.items()
            if k.startswith("curve.")
        }
        obj.curve = ExpKModel.load(curve_cfg)
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
    if T == 0:
        return 0
    if delta / T < 60:
        p = np.exp(-delta / T)
    else:
        p = 0
    return p


def binary_search(
    func: Callable[[NDArray], NDArray],
    target: float,
    x_min: float,
    x_max: float,
    epsilon: float,
    f_epsilon: Optional[float] = None,
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


def generate_random_integer_array(n, K):
    # Generate n-1 random points between 0 and K
    random_points = np.sort(np.random.randint(0, K, n - 1))

    # Add the end points (0 and K) to the list of points
    random_points = np.concatenate(([0], random_points, [K]))

    # The differences between consecutive points are the random segments
    random_segments = np.diff(random_points)

    return random_segments
