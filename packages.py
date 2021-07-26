""""""
import numpy as np


class CreatePackage:
    def __init__(self, n_packages, arrival_rate):
        self.n_packages = n_packages
        self.arrival_rate = arrival_rate
        self.packages = [Package(i, self.arrival_time()[i]) for i in range(self.n_packages)]

    def arrival_time(self):
        r = np.random.exponential(1/self.arrival_rate, size=self.n_packages)
        t = np.hstack([[0], np.cumsum(r)])
        return t


class Package:
    def __init__(self, i, t):
        self.id = i
        self.arriving_time = t
        self.halt_time = 0
        self.sending_time = None
        self.target = None
        self.serving_time = None


if __name__ == '__main__':
    package = CreatePackage(100, 10)
    print(package.arrival_time())
