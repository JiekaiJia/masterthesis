""""""
import numpy as np


class Package:
    def __init__(self, n_jobs, arrival_rate):
        self.n_jobs = n_jobs
        self.arrival_rate = arrival_rate
        self.arriving_time = self.arrival_time()

    def arrival_time(self):
        r = np.random.exponential(1/self.arrival_rate, size=self.n_jobs)
        t = np.hstack([[0], np.cumsum(r)])
        return t


if __name__ == '__main__':
    package = Package(100, 100)
    print(package.arriving_time)
