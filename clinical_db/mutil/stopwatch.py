import time


class Stopwatch:
    """Stopwatch class for measureing time."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.base_cpu = time.clock()
        self.base_real = time.time()

    def stop(self):
        self.end_cpu = time.clock()
        self.end_real = time.time()

    def cpu_elapsed(self):
        return self.end_cpu - self.base_cpu

    def real_elapsed(self):
        return self.end_real - self.base_real

    def print_cpu_elapsed(self, stop=False):
        if stop:
            self.stop()
        print "CPU Elapsed:%0.4f[sec]" % self.cpu_elapsed()

    def print_real_elapsed(self, stop=False):
        if stop:
            self.stop()
        print "Real Elapsed:%0.4f[sec]" % self.real_elapsed()
