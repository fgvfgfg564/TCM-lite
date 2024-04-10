from typing_extensions import TypeVar

import asyncio
import concurrent
from concurrent.futures import ProcessPoolExecutor
from typing_extensions import List, Callable

T1 = TypeVar("T1")
T2 = TypeVar("T2")


def async_map(
    f: Callable[[T1], T2], x: List[T1], executor: ProcessPoolExecutor
) -> List[T2]:
    """
    A parallel mapping function that safe to Ctrl-C kill
    """

    async def _async_map() -> List[T2]:
        loop = asyncio.get_running_loop()
        tasks: List[asyncio.futures.Future[T2]] = []
        for itm in x:
            task = loop.run_in_executor(executor, f, itm)
            tasks.append(task)

        try:
            results = await asyncio.gather(*tasks)
        except (KeyboardInterrupt, asyncio.CancelledError) as e:
            for task in tasks:
                task.cancel()
            for process in executor._processes.values():
                process.terminate()
            executor.shutdown(wait=False, cancel_futures=True)
            await asyncio.gather(*tasks, return_exceptions=True)
            raise e
        return results

    return asyncio.run(_async_map())
