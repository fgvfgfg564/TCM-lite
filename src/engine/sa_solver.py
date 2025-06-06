from typing import Optional, cast
from numpy import ndarray
from sympy import NDimArray
from typing_extensions import Tuple, List, Iterable, override

import abc
import os
import copy
from functools import partial
from scipy.optimize import minimize, newton
from concurrent.futures import ProcessPoolExecutor
from time import time

from ..loss import LOSSES

from ..async_ops import async_map
from ..type import *
from ..fileio import FileIO
from ..math_utils import *


class SolverBase(abc.ABC):

    def _loss(
        self,
        precomputed_curve: ImgCurves,
        n_ctu,
        file_io: FileIO,
        method_ids: NDArray,
        target_byteses: NDArray,
        losstype: str,
        r_limit: Optional[float] = None,
        t_limit: Optional[float] = None,
    ) -> Tuple[LossType, float, float]:
        # Returns score given method ids and target bytes
        if r_limit is not None and np.sum(target_byteses) > r_limit:
            r_exceed = np.sum(target_byteses) - r_limit
        else:
            r_exceed = 0

        ctu_losses = []
        global_time = 0

        for i in range(n_ctu):
            method_id: int = method_ids[i]
            target_bytes: float = target_byteses[i]

            precomputed_results = precomputed_curve[method_id][i]

            ctu_loss: NDArray[np.float32] = precomputed_results["b_e"](target_bytes)
            t = np.polyval(precomputed_results["b_t"], target_bytes)

            global_time += t
            ctu_losses.append(ctu_loss)

        losscls = LOSSES[losstype]
        dloss = losscls.global_level_loss(file_io, ctu_losses)
        if t_limit is not None and global_time > t_limit:
            t_exceed = global_time - t_limit
        else:
            t_exceed = 0
        loss = LossType(r=r_exceed, t=t_exceed, d=dloss)
        return loss, dloss, global_time

    @abc.abstractmethod
    def find_optimal_target_bytes(
        self,
        precomputed_curve: ImgCurves,
        file_io: FileIO,
        n_ctu: int,
        method_ids: ndarray,
        r_limit: float,
        t_limit: float,
        losstype: str,
    ) -> Tuple[np.ndarray, LossType, float, float]:
        pass


# class SLSQPSolver(SolverBase):
#     @staticmethod
#     def _calc_gradient_psnr(sqes: np.ndarray):
#         r"""
#         $ PSNR(sqes) = - 10 * \log_{10}{(\frac{\sum X}{num\_pixels})} $
#         """

#         return -10 / (sqes.sum() * np.log(10))

#     @override
#     def find_optimal_target_bytes(
#         self,
#         precomputed_curve: ImgCurves,
#         file_io: FileIO,
#         n_ctu,
#         method_ids: Iterable[int],
#         r_limit: float,
#         t_limit: float,
#         learning_rate: float = 1e3,
#         num_steps: int = 1000,
#         init_value=None,
#     ) -> Tuple[np.ndarray, LossType, float, float]:
#         """
#         Find the optimal target bytes given CTU methods
#         """

#         min_bytes = np.zeros(
#             [
#                 n_ctu,
#             ],
#             dtype=np.float32,
#         )
#         max_bytes = np.zeros(
#             [
#                 n_ctu,
#             ],
#             dtype=np.float32,
#         )

#         bounds = []

#         for i in range(n_ctu):
#             min_bytes[i] = precomputed_curve[method_ids[i]][i]["b_e"].X_min
#             max_bytes[i] = precomputed_curve[method_ids[i]][i]["b_e"].X_max
#             bounds.append((min_bytes[i], max_bytes[i]))

#         if init_value is None:
#             bpp = r_limit / file_io.num_pixels * 0.99
#             init_value = file_io.block_num_pixels * bpp

#         init_value = normalize_to_target(init_value, min_bytes, max_bytes, r_limit)

#         if init_value.sum() > r_limit:
#             score = -float("inf")
#             psnr = score
#             time = -score
#             return init_value, score, psnr, time

#         def objective_func(target_bytes) -> LossType:
#             result = self._loss(
#                 precomputed_curve,
#                 n_ctu,
#                 file_io,
#                 method_ids,
#                 target_bytes,
#             )[0]
#             return (result[0], result[1], result[2] * learning_rate)

#         def grad(target_bytes):
#             gradients = np.zeros_like(target_bytes)

#             # Gradient item on sqe
#             sqes = []
#             for i in range(n_ctu):
#                 method_id = method_ids[i]
#                 precomputed_results = precomputed_curve[method_id][i]
#                 sqes.append(precomputed_results["b_e"](target_bytes[i]))
#             sqes = np.asarray(sqes)

#             sqe_gradient = self._calc_gradient_psnr(np.array(sqes))

#             gradients = []
#             for i in range(n_ctu):
#                 method_id = method_ids[i]
#                 b_e = precomputed_curve[method_id][i]["b_e"]
#                 b_t = precomputed_curve[method_id][i]["b_t"]
#                 gradients.append(
#                     self.w_time * b_t[0]
#                     - sqe_gradient * b_e.derivative(target_bytes[i])
#                 )

#             return np.asarray(gradients) * learning_rate

#         def ineq_constraint_r(target_bytes):
#             return r_limit - target_bytes.sum()

#         def ineq_constraint_t(target_bytes):
#             _, __, t = self._loss(
#                 precomputed_curve,
#                 n_ctu,
#                 file_io,
#                 method_ids,
#                 target_bytes,
#             )
#             return t_limit - t

#         constraint_r = {"type": "ineq", "fun": ineq_constraint_r}
#         constraint_t = {"type": "ineq", "fun": ineq_constraint_t}

#         result = minimize(
#             objective_func,
#             init_value,
#             jac=grad,
#             method="SLSQP",
#             bounds=bounds,
#             constraints=[constraint_r, constraint_t],
#             options={
#                 "ftol": 1e-12,
#                 "maxiter": num_steps,
#             },
#         )

#         ans = result.x

#         score, psnr, time = self._loss(n_ctu, file_io, method_ids, ans, b_t)

#         return ans, score, psnr, time


class LagrangeMultiplierSolver(SolverBase):
    def __init__(self, num_workers: WorkerConfig = "AUTO") -> None:
        super().__init__()
        if num_workers == "AUTO":
            cpu_count = os.cpu_count()
            if cpu_count is not None:
                self.num_workers = min(cpu_count * 2, 32)
            else:
                self.num_workers = 32
        else:
            self.num_workers = num_workers
        self.batch_size = 4
        self.PPE = ProcessPoolExecutor(max_workers=self.num_workers)

    @classmethod
    def _bs_inner_loop(cls, target_d: float, curves: List[Fitter]) -> List[float]:
        roots = []
        for curve in curves:
            fdx = curve.curve.derivative()
            fdx = copy.copy(fdx)
            fdx.c -= target_d
            if fdx(curve.X_min) > 0:
                # fdx should be increasing
                roots.append(curve.X_min)
            elif fdx(curve.X_max) < 0:
                roots.append(curve.X_max)
            else:
                try:
                    root = newton(
                        fdx, x0=0.0, fprime=fdx.derivative(), maxiter=50, tol=1.0
                    )
                    root = cast(NDArray, root)
                except RuntimeError:
                    print(
                        f"WARNING: Newton's method failed. Curve={curve.curve}; Derivative func={fdx}"
                    )
                    root = binary_search(
                        fdx, 0.0, x_min=curve.X_min, x_max=curve.X_max, epsilon=1.0
                    )
                root = np.clip(root, curve.X_min, curve.X_max)
                roots.append(root)
        return roots

    @classmethod
    def separate_into_sublists(cls, lst, length):
        return list([lst[i : i + length] for i in range(0, len(lst), length)])

    @classmethod
    def join_lists(cls, lists):
        return list([element for sublist in lists for element in sublist])

    def get_ctu_results(self, curves_list, target_d: float, method_ids, n_ctu):
        _f = partial(self._bs_inner_loop, target_d)

        curves_list = self.separate_into_sublists(curves_list, self.batch_size)
        ctu_results = async_map(_f, curves_list, executor=self.PPE)
        return self.join_lists(ctu_results)

    # Actually, this step can basically ignore T loss.
    def find_optimal_target_bytes(
        self,
        precomputed_curve: ImgCurves,
        file_io: FileIO,
        n_ctu,
        method_ids,
        r_limit,
        t_limit,
        losstype,
    ):
        # A simple Lagrange Multiplier
        curves_list = list(
            [precomputed_curve[method_ids[i]][i]["b_e"] for i in range(n_ctu)]
        )

        def outer_loop(target_d: float) -> float:
            ctu_results = self.get_ctu_results(curves_list, target_d, method_ids, n_ctu)
            tot_bytes = sum(ctu_results)
            return tot_bytes

        min_d = float("inf")
        max_d = -float("inf")

        for curve in curves_list:
            min_do = curve.derivative(curve.X_min)
            max_do = curve.derivative(curve.X_max)
            min_d = min(min_d, min_do)
            max_d = max(max_d, max_do)

        # if outer_loop(min_d) >= r_limit - 4:
        #     target_d = min_d
        # elif outer_loop(max_d) <= r_limit - 4:
        #     target_d = max_d
        # else:
        #     # Zero-point exists
        #     target_d = brentq(
        #         lambda x: outer_loop(x) - r_limit - 4,
        #         min_d,
        #         max_d,
        #         xtol=1e-6,
        #         rtol=1e-4,
        #     )
        target_d = binary_search(
            outer_loop, r_limit, min_d, max_d, epsilon=1e-6, f_epsilon=32
        )
        ctu_results = self.get_ctu_results(curves_list, target_d, method_ids, n_ctu)
        loss, psnr, t = self._loss(
            precomputed_curve=precomputed_curve,
            n_ctu=n_ctu,
            file_io=file_io,
            method_ids=method_ids,
            target_byteses=ctu_results,
            r_limit=r_limit,
            t_limit=t_limit,
            losstype=losstype,
        )
        return np.asarray(ctu_results), loss, psnr, t
