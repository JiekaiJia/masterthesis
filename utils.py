"""https://github.com/minqi/learning-to-communicate-pytorch"""
import bisect
import copy
import math


class DotDic(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        return DotDic(copy.deepcopy(dict(self), memo=memo))


def index(a, x):
    """Get the position of x, when inserting x into a."""
    i = bisect.bisect_left(a, x)
    if i != len(a):
        return i
    return i-1


def sigmoid(x):
    """x is a scalar, if want to extend to vector use map. return a list."""
    if len(x) > 1:
        return [1/(1 + math.exp(-_x)) for _x in x]
    return [1/(1 + math.exp(-x))]
