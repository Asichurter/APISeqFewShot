import time
from functools import wraps

def costTimeWithFuncName(func_name):
    def cost_time(func):
        def real_func(*args, **kwargs):
            start = time.time()
            ret = func(*args, **kwargs)
            end = time.time()
            print(f"[{func_name}] cost time: %.5f"%(end-start))
            return ret
        return real_func
    return cost_time


class FuncProfiler:
    def __init__(self, name, report_iter=None):
        self.Name = name
        self.Iter = {'Train': 0, 'Validate': 0, 'Test': 0}
        self.Total = {'Train': 0, 'Validate': 0, 'Test': 0}
        self.ReportIter = {'Train': 200,
                           'Validate': 100,
                           'Test': 100}
        if report_iter is not None:
            self.ReportIter = report_iter

    def _clear(self):
        self.Total = {'Train': 0, 'Validate': 0, 'Test': 0}

    def __call__(self, func):

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            task_type = kwargs.get('task_type', None)
            if task_type is None:
                return func(*args, **kwargs)

            self.Iter[task_type] += 1

            start = time.time()
            res = func(*args, **kwargs)
            end = time.time()
            self.Total[task_type] += (end - start)

            iters = self.Iter[task_type]
            report_iter = self.ReportIter[task_type]

            if iters % report_iter == 0:
                print(f"- {task_type}.{self.Name}: " + "%.5f" % self.Total[task_type])
                self._clear()

            return res

        return wrapped_func


