import functools
from typing import Callable
from time import perf_counter_ns


def timeit() -> Callable:
    """
    Decorator for timing function execution time.
    """

    def _timeit(func: Callable) -> Callable:
        @functools.wraps(func)
        def timed(*args, **kw):
            start = perf_counter_ns()
            result = func(*args, **kw)
            end = perf_counter_ns()

            total_sec = int((end - start) / 1e9)  # nanosec to sec
            total_ms = int((end - start) / 1e6)  # nanosec to ms
            time_string = f"{total_sec} seconds" if total_sec > 10 else f"{total_ms} ms"
            message = f"{func.__name__} Execution time: {time_string}."

            print(message)

            return result

        return timed

    return _timeit