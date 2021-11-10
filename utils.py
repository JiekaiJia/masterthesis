"""https://github.com/minqi/learning-to-communicate-pytorch"""
import bisect
import copy


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
    raise ValueError
