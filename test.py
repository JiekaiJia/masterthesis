import json
import time

import tqdm

from custom_env.environment import MainEnv
from dotdic import DotDic
import policies

if __name__ == '__main__':
    start = time.time()

    with open('./config/PartialAccess.json', 'r') as f:
        config = DotDic(json.loads(f.read()))
    config.training = False
    config.use_belief = False
    config.silent = True
    # Initialize Policy
    policy = policies.RandomPolicy(config)
    # Initialize Environment
    env = MainEnv(config)
    acc_r = 0
    steps = 0
    drop_pkg = {scheduler: 0 for scheduler in env.schedulers}
    print(f'{policy.__class__.__name__} start running!!')
    for _ in tqdm.tqdm(range(120)):
        obss = env.reset()
        step = 0
        dones = {'__all__': False}
        while not dones['__all__']:
            # Random policy
            actions = {scheduler: policy.get_actions(obs) for scheduler, obs in obss.items()}

            obss, r, dones, info = env.step(actions)
            obss = {scheduler: obs[0] for scheduler, obs in obss.items()}
            acc_r += sum(r.values())
            # print('timestep:', step + 1)
            # print(obss)
            # print('_' * 80)
            step += 1
        for k, v in env.acc_drop_pkgs.items():
            drop_pkg[k] += v
        steps += step
    end = time.time()

    print(f'mean episode rewards: {acc_r/120:.2f}')
    print(f'mean episode length: {steps/120:.2f}')
    print('mean dropped packages rate:', {k: round(v/(120*config.n_packages), 2) for k, v in drop_pkg.items()})
    print(f'Runtime: {end-start}S')