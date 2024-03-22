import pyamosa, numpy as np


class ZDT1(pyamosa.Problem):
    n_var = 30

    def __init__(self):

        pyamosa.Problem.__init__(
            self,
            ZDT1.n_var,
            [pyamosa.Type.REAL] * ZDT1.n_var,
            [0.0] * ZDT1.n_var,
            [1.0] * ZDT1.n_var,
            2,
            0,
        )

    def evaluate(self, x, out):
        f = x[0]
        g = 1 + 9 * sum(x[1:]) / (self.num_of_variables - 1)
        h = 1 - np.sqrt(f / g)
        out["f"] = [f, g * h]


if __name__ == "__main__":
    problem = ZDT1()

    config = pyamosa.Config()
    config.archive_hard_limit = 100
    config.archive_soft_limit = 500
    config.archive_gamma = 2
    config.clustering_max_iterations = 300
    config.hill_climbing_iterations = 500
    config.initial_temperature = 1.0
    config.cooling_factor = 0.9
    config.annealing_iterations = 1000
    config.annealing_strength = 1
    config.multiprocess_enabled = True

    optimizer = pyamosa.Optimizer(config=config)

    termination = pyamosa.StopMinTemperature(1e-8)
    optimizer.run(problem, termination)

    print(optimizer.archive.candidate_solutions)
    optimizer.archive.write_json("./tmp/amosa.json")
    optimizer.archive.plot_front(problem, "./tmp/plot.pdf")
