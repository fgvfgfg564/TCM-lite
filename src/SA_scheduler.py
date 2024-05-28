import abc
import time


class BaseScheduler(abc.ABC):
    def start(self) -> None:
        """
        Terminator start
        """
        pass

    @abc.abstractmethod
    def next(self) -> None:
        """
        Next SA step
        """

    start_T = 0.0

    @property
    @abc.abstractmethod
    def step_size(self) -> float:
        """
        Step size of SA algorithm
        """

    @abc.abstractmethod
    def should_stop(self) -> bool:
        """
        Whether to stop the SA algorithm
        """


class NStepsScheduler(BaseScheduler):
    def __init__(self, start_T, end_T, num_steps):
        """
        Terminator for SA algorithm with fixed step size
        :param step_size: Step size of SA algorithm
        """
        self.start_T = start_T
        self.end_T = end_T
        self.num_steps = num_steps
        self.step = 0

    def start(self) -> None:
        self.step = 0

    @property
    def step_size(self):
        return (self.end_T / self.start_T) ** (1 / self.num_steps)

    def next(self):
        self.step += 1

    def should_stop(self):
        return self.step >= self.num_steps

    def __repr__(self) -> str:
        return f"<NStepsScheduler {self.__dict__}>"


class TimeLapseScheduler(BaseScheduler):
    def __init__(self, start_T, step_size, max_time):
        """
        Terminator for SA algorithm with fixed step size
        :param step_size: Step size of SA algorithm
        """
        self.start_T = start_T
        self._step_size = step_size
        self.max_time = max_time
        self.start_time = time.perf_counter()

    def start(self):
        self.start_time = time.perf_counter()

    def next(self):
        pass

    @property
    def step_size(self):
        return self._step_size

    def should_stop(self):
        return time.perf_counter() - self.start_time > self.max_time

    def __repr__(self) -> str:
        return f"<TimeLapseScheduler {self.__dict__}>"
