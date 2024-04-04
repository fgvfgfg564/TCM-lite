from typing_extensions import *

import abc
import os
from functools import partial
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor
from time import time

from ..async_ops import async_map
from ..type import *
from ..fileio import FileIO
from ..math_utils import *


class SolverBase(abc.ABC):
    def _get_score(
        self,
        precomputed_curve: ImgCurves,
        n_ctu,
        file_io: FileIO,
        method_ids,
        target_byteses,
        r_limit=None,
        t_limit=None,
    ) -> Tuple[LossType, float, float]:
        # Returns score given method ids and target bytes
        if r_limit is not None and np.sum(target_byteses) > r_limit:
            r_exceed = np.sum(target_byteses) > r_limit
        else:
            r_exceed = 0

        sqe = 0
        global_time = 0

        for i in range(n_ctu):
            method_id = method_ids[i]
            target_bytes = target_byteses[i]

            precomputed_results = precomputed_curve[method_id][i]

            ctu_sqe = precomputed_results["b_e"](target_bytes)
            t = np.polyval(precomputed_results["b_t"], target_bytes)

            global_time += t
            sqe += ctu_sqe

        sqe /= file_io.num_pixels * 3
        psnr = -10 * np.log10(sqe)
        if t_limit is not None and global_time > t_limit:
            t_exceed = global_time - t_limit
        else:
            t_exceed = 0
        loss = LossType((r_exceed, t_exceed, -psnr))
        return loss, psnr, global_time

    @abc.abstractmethod
    def find_optimal_target_bytes(
        self,
        precomputed_curve: ImgCurves,
        file_io: FileIO,
        n_ctu,
        method_ids,
        r_limit,
        t_limit,
    ) -> Tuple[np.ndarray, LossType, float, float]:
        pass


class SLSQPSolver(SolverBase):
    @staticmethod
    def _calc_gradient_psnr(sqes: np.ndarray):
        r"""
        $ PSNR(sqes) = - 10 * \log_{10}{(\frac{\sum X}{num\_pixels})} $
        """

        return -10 / (sqes.sum() * np.log(10))

    def find_optimal_target_bytes(
        self,
        precomputed_curve: ImgCurves,
        file_io: FileIO,
        n_ctu,
        method_ids: Iterable[int],
        r_limit,
        t_limit,
        learning_rate=1e3,
        num_steps=1000,
        init_value=None,
    ):
        """
        Find the optimal target bytes given CTU methods
        """

        min_bytes = np.zeros(
            [
                n_ctu,
            ],
            dtype=np.int32,
        )
        max_bytes = np.zeros(
            [
                n_ctu,
            ],
            dtype=np.int32,
        )

        bounds = []

        for i in range(n_ctu):
            min_bytes[i] = precomputed_curve[method_ids[i]][i]["b_e"].X_min
            max_bytes[i] = precomputed_curve[method_ids[i]][i]["b_e"].X_max
            bounds.append((min_bytes[i], max_bytes[i]))

        if init_value is None:
            bpp = r_limit / file_io.num_pixels * 0.99
            init_value = file_io.block_num_pixels * bpp

        init_value = normalize_to_target(init_value, min_bytes, max_bytes, r_limit)

        if init_value.sum() > r_limit:
            score = -float("inf")
            psnr = score
            time = -score
            return init_value, score, psnr, time

        def objective_func(target_bytes) -> LossType:
            result = self._get_score(
                precomputed_curve,
                n_ctu,
                file_io,
                method_ids,
                target_bytes,
            )[0]
            return (result[0], result[1], result[2] * learning_rate)

        def grad(target_bytes):
            gradients = np.zeros_like(target_bytes)

            # Gradient item on sqe
            sqes = []
            for i in range(n_ctu):
                method_id = method_ids[i]
                precomputed_results = precomputed_curve[method_id][i]
                sqes.append(precomputed_results["b_e"](target_bytes[i]))
            sqes = np.asarray(sqes)

            sqe_gradient = self._calc_gradient_psnr(np.array(sqes))

            gradients = []
            for i in range(n_ctu):
                method_id = method_ids[i]
                b_e = precomputed_curve[method_id][i]["b_e"]
                b_t = precomputed_curve[method_id][i]["b_t"]
                gradients.append(
                    self.w_time * b_t[0]
                    - sqe_gradient * b_e.derivative(target_bytes[i])
                )

            return np.asarray(gradients) * learning_rate

        def ineq_constraint_r(target_bytes):
            return r_limit - target_bytes.sum()

        def ineq_constraint_t(target_bytes):
            _, __, t = self._get_score(
                precomputed_curve,
                n_ctu,
                file_io,
                method_ids,
                target_bytes,
            )
            return t_limit - t

        constraint_r = {"type": "ineq", "fun": ineq_constraint_r}
        constraint_t = {"type": "ineq", "fun": ineq_constraint_t}

        result = minimize(
            objective_func,
            init_value,
            jac=grad,
            method="SLSQP",
            bounds=bounds,
            constraints=[constraint_r, constraint_t],
            options={
                "ftol": 1e-12,
                "maxiter": num_steps,
            },
        )

        ans = result.x

        score, psnr, time = self._get_score(n_ctu, file_io, method_ids, ans, b_t)

        return ans, score, psnr, time


class LagrangeMultiplierSolver(SolverBase):
    def __init__(self, num_workers: WorkerConfig = "AUTO") -> None:
        super().__init__()
        if num_workers == "AUTO":
            self.num_workers = os.cpu_count() * 2
        else:
            self.num_workers = num_workers
        self.batch_size = 32
        self.PPE = ProcessPoolExecutor(max_workers=self.num_workers)

    @classmethod
    def _bs_inner_loop(cls, target_d: float, curves: List[Fitter]) -> float:
        roots = []
        for curve in curves:
            fdx = curve.curve.derivative()
            fdx[-1] -= target_d
            raise NotImplemented  # TODO
            root = fdx.posroot()
            root = np.clip(root, curve.X_min, curve.X_max)
            roots.append(root)
        return roots

    @classmethod
    def separate_into_sublists(cls, lst, length):
        return [lst[i : i + length] for i in range(0, len(lst), length)]

    @classmethod
    def join_lists(cls, lists):
        return list([element for sublist in lists for element in sublist])

    def get_ctu_results(self, precomputed_curve, target_d: float, method_ids, n_ctu):
        _f = partial(self._bs_inner_loop, target_d)

        curves_list = list(
            [precomputed_curve[method_ids[i]][i]["b_e"] for i in range(n_ctu)]
        )
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
    ):
        # A simple Lagrange Multiplier

        def outer_loop(target_d: float) -> float:
            ctu_results = self.get_ctu_results(
                precomputed_curve, target_d, method_ids, n_ctu
            )
            tot_bytes = sum(ctu_results)
            return tot_bytes

        target_d = binary_search(outer_loop, r_limit, -1, 0, 1e-6)
        ctu_results = self.get_ctu_results(
            precomputed_curve, target_d, method_ids, n_ctu
        )
        score, psnr, t = self._get_score(
            precomputed_curve=precomputed_curve,
            n_ctu=n_ctu,
            file_io=file_io,
            method_ids=method_ids,
            target_byteses=ctu_results,
            r_limit=r_limit,
            t_limit=t_limit,
        )
        return np.asarray(ctu_results), score, psnr, t
