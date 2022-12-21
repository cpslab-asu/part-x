import time

class Fn:
    def __init__(self, func):
        self.func = func
        self.count = 0
        self.point_history = []
        self.simultation_time = []

    def __call__(self, *args, **kwargs):
        self.count = self.count + 1
        sim_time_start = time.perf_counter()
        rob_val = self.func(*args, **kwargs)
        time_elapsed = time.perf_counter() - sim_time_start
        self.simultation_time.append(time_elapsed)
        self.point_history.append([self.count, *args, rob_val])
        return rob_val
