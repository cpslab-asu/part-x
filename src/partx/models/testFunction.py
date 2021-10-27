class callCounter:
    def __init__(self,func):
        self.func = func
        self.callCount = 0
        self.point_history = []
    def __call__(self, *args, **kwargs):
        self.callCount = self.callCount + 1
        rob_val = self.func(*args, **kwargs)
        self.point_history.append([self.callCount, *args, rob_val])
        return rob_val
