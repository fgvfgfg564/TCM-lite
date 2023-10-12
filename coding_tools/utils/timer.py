import time
import torch

__all__ = ["timing_decorator", "Timer"]

LEVEL = 0
ENABLE = True


def timing_decorator(name):
    """
    Decorator to measure the execution time of a function anytime it's invoked if ENABLE=True
    """

    def _decorator(func):
        if ENABLE:

            def wrapper(*args, **kwargs):
                global LEVEL
                torch.cuda.synchronize()
                start_time = time.time()
                LEVEL += 1
                result = func(*args, **kwargs)
                LEVEL -= 1
                torch.cuda.synchronize()
                end_time = time.time()
                execution_time = end_time - start_time
                print(
                    "\t" * LEVEL + f"{name} took {execution_time} seconds to execute."
                )
                return result

            return wrapper
        else:
            return func

    return _decorator


class Timer:
    """
    Context manager to measure the execution time of a piece of code if ENABLE=True
    """

    def __init__(self, name) -> None:
        self.time_start = None
        self.name = name

    def __enter__(self):
        if ENABLE:
            torch.cuda.synchronize()
            self.time_start = time.time()
            global LEVEL
            LEVEL += 1

    def __exit__(self, exc_type, exc_value, traceback):
        if ENABLE:
            torch.cuda.synchronize()
            global LEVEL
            LEVEL -= 1
            print(
                "\t" * LEVEL
                + f"{self.name} took {time.time() - self.time_start} seconds to execute."
            )
