import numpy as np
import time


def report_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        print(f"Function {func.__name__} took {duration_ms:.3f} milliseconds to run.")
        return result

    return wrapper
