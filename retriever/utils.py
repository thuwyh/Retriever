from time import perf_counter
import logging

def timer(logger:logging.Logger=None):
    def _timer(func):
        def wrapper(*args, **kwargs):
            start_time = perf_counter()
            result = func(*args, **kwargs)
            end_time = perf_counter()
            logger.info(f"Finish '{func.__name__}' in {end_time - start_time:.3f} seconds.")
            return result
        return wrapper
    return _timer