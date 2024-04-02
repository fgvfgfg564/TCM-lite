import asyncio
import concurrent
from concurrent.futures import ProcessPoolExecutor
from typing_extensions import List, Callable


def async_map(
    f: Callable[[float], float], x: List[float], executor: ProcessPoolExecutor
) -> List[float]:
    """
    A parallel mapping function that safe to Ctrl-C kill
    """

    async def _async_map() -> List[float]:
        loop = asyncio.get_running_loop()
        tasks: List[concurrent.futures.Future] = []
        for itm in x:
            task = loop.run_in_executor(executor, f, itm)
            tasks.append(task)

        try:
            results = await asyncio.gather(*tasks)
        except (KeyboardInterrupt, asyncio.CancelledError):
            for task in tasks:
                task.cancel()
            for process in executor._processes.values():
                process.terminate()
            executor.shutdown(wait=False, cancel_futures=True)
            await asyncio.gather(*tasks, return_exceptions=True)
        return results

    return asyncio.run(_async_map())
