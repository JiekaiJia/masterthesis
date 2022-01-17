"""https://github.com/minqi/learning-to-communicate-pytorch"""
import bisect
import copy
import math

from scipy.special import softmax


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
    try:
        if len(x) > 1:
            return [1/(1 + math.exp(-_x)) for _x in x]
    except TypeError:
        pass
    return [1/(1 + math.exp(-x))]


if __name__ == '__main__':
    import json
    from custom_env.environment import SuperObsEnv
    with open('./config/PartialAccess.json', 'r') as f:
        config = DotDic(json.loads(f.read()))
    config.training = False
    config.use_belief = False
    config.silent = True
    env = SuperObsEnv(config)
    obs = env.reset()
    print(env.obs_mask)
    for scheduler in env.schedulers:
        print(scheduler, env.observation_spaces[scheduler].contains(obs[scheduler]))
    actions = {scheduler: env.action_spaces[scheduler].sample() for scheduler in env.schedulers}
    obs, r, dones, info = env.step(actions)
    for scheduler in env.schedulers:
        print(scheduler, env.observation_spaces[scheduler].contains(obs[scheduler]))
        print(obs[scheduler])
    # step = 0
    # print(env.observation_spaces)
    # print('step', step)
    # print('obs', obs)
    # print('-' * 40)
    # done = False
    # dones = {'__all__': False}
    # while not dones['__all__']:
    #     actions = {scheduler: env.action_spaces[scheduler].sample() for scheduler in env.schedulers}
    #     obs, r, dones, info = env.step(actions)
    #     step += 1
    #     print('step', step)
    #     print('action', actions)
    #     print('obs', obs)
    #     print('r', r)
    #     print('info', info)
    #     print('-'*40)
