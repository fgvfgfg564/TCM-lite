import abc
import os
from functools import partial

from ..async_ops import async_map
from ..type import *
from ..fileio import FileIO
from ..math_utils import binary_search


class SolverBase(abc.ABC):
    def _get_score(
        self,
        precomputed_curve: ImgCurves,
        n_ctu,
        file_io: FileIO,
        method_ids,
        target_byteses,
        r_limit,
        w_time=None,
        time_limit=None,
    ):
        # Returns score given method ids and target bytes
        if np.sum(target_byteses) > r_limit:
            return -np.inf, -np.inf, np.inf

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
        if time_limit is None:
            return psnr - w_time * global_time, psnr, global_time
        else:
            if global_time > time_limit:
                return -np.inf, -np.inf, np.inf
            return psnr, psnr, global_time

    @abc.abstractmethod
    def find_optimal_target_bytes(
        self, precomputed_curve: ImgCurves, file_io: FileIO, n_ctu, method_ids, b_t
    ):
        pass


class LagrangeMultiplierSolver(SolverBase):
    def __init__(self, num_workers: WorkerConfig = "AUTO") -> None:
        super().__init__()
        if num_workers == "AUTO":
            self.num_workers = min(os.cpu_count() * 2, 24)
        else:
            self.num_workers = num_workers

    @classmethod
    def _bs_inner_loop(cls, target_d: float, curve: Warpped4DFitter) -> float:
        fdx = curve.curve.derivative().copy()
        fdx[-1] -= target_d
        root = fdx.posroot()
        root = np.clip(root, curve.X_min, curve.X_max)
        return root

    def get_ctu_results(self, target_d: float, method_ids, n_ctu):
        _f = partial(self._bs_inner_loop, target_d)

        curves_list = list(
            [self._precomputed_curve[method_ids[i]][i]["b_e"] for i in range(n_ctu)]
        )
        ctu_results = async_map(_f, curves_list, num_workers=self.num_workers)
        return ctu_results

    # Actually, this step can basically ignore T loss.
    def find_optimal_target_bytes(
        self,
        precomputed_curve: ImgCurves,
        file_io: FileIO,
        n_ctu,
        method_ids,
        b_t,
    ):
        # A simple Lagrange Multiplier
        self._precomputed_curve = precomputed_curve

        def outer_loop(target_d: float) -> float:
            ctu_results = self.get_ctu_results(target_d, method_ids, n_ctu)
            tot_bytes = sum(ctu_results)
            return tot_bytes

        target_d = binary_search(outer_loop, b_t, -1, 0, 1e-6)
        ctu_results = self.get_ctu_results(target_d, method_ids, n_ctu)
        score, psnr, t = self._get_score(
            n_ctu=n_ctu,
            file_io=file_io,
            method_ids=method_ids,
            target_byteses=ctu_results,
            r_limit=b_t,
        )
        return 
