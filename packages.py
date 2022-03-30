"""This module provides the package classes."""
import numpy as np


class CreatePackage:
    """
    n_packages: The number of packages,
    arrival_rate: How many packages would arrive per unit time,
    sender: The scheduler's name who send this packages.
    """
    def __init__(self, n_packages, arrival_rate, sender):
        self.n_packages = n_packages
        self.arrival_rate = arrival_rate
        self.arrival_time = self.arrival_time()
        self.packages = [Package(i, self.arrival_time[i], sender) for i in range(self.n_packages)]

    def arrival_time(self):
        # The arrival time satisfies Possion distribution, where time interval satisfies Exponential distribution.
        r = np.random.exponential(1/self.arrival_rate, size=self.n_packages)
        t = np.hstack([[0], np.cumsum(r)])
        return t


class Package:
    def __init__(self, i, t, sender):
        self.id = i
        self.target = None
        self.sender = sender
        # The time a package arrives at scheduler's queue.
        self.arriving_time = t
        # The time a package is sent from scheduler's queue.
        self.sending_time = None
        # The time a package is served.
        self.serving_time = None
        # The time a package departures the server.
        self.departure_time = None


if __name__ == '__main__':
    package = CreatePackage(100, 0.9, 1)
    print(package.arrival_time)
