import json
import time

import tqdm

from custom_env.environment import MainEnv
from utils import DotDic
import policies

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--act_frequency', type=int, nargs='+', default=0, help="gives actions frequency [default: 0]")
    args = parser.parse_args()

    start = time.time()

    with open('./config/PartialAccess.json', 'r') as f:
        config = DotDic(json.loads(f.read()))
    config.training = False
    config.use_belief = False
    config.silent = True
    mean_episode_rewards, mean_episode_length, total_drp_pkg_rate = [], [], []
    # act_frequencies = args.act_frequency
    act_frequencies = range(1)
    for act_frequency in act_frequencies:
        config.act_frequency = act_frequency
        # Initialize Environment
        env = MainEnv(config)
        # Initialize Policy
        policy = policies.RandomPolicy(config, env)
        print(f'{policy.__class__.__name__} start running!!')
        acc_r, steps = 0, 0
        num_e = 1
        drop_pkg = {scheduler: 0 for scheduler in env.schedulers}
        for _ in tqdm.tqdm(range(num_e)):
            obss = env.reset()
            real_obss = obss
            step = 0
            dones = {'__all__': False}
            while not dones['__all__']:
                # print('timestep:', step + 1)
                actions = {scheduler: policy.get_gactions(obs) for scheduler, obs in obss.items()}
                obss, r, dones, info = env.step(actions)
                _obss = obss
                obss = {scheduler: obs[0] for scheduler, obs in _obss.items()}
                real_obss = {scheduler: obs[1] for scheduler, obs in _obss.items()}
                acc_r += sum(r.values())
                # print('scheduler state', env.schedulers_state())
                # print('env state', env.state())
                # print('obss', obss)
                # print('real_obss', real_obss)
                # print('reward', r)
                # print(env.acc_drop_pkgs)
                # print('_' * 80)
                step += 1
            if env.acc_drop_pkgs[env.schedulers[0]] == 60:
                break
            for k, v in env.acc_drop_pkgs.items():
                drop_pkg[k] += v
            steps += step
        end = time.time()
        drp_pkg_rate = {k: round(v / (num_e * config.n_packages), 2) for k, v in drop_pkg.items()}
        mean_episode_length.append(round(steps/num_e, 2))
        mean_episode_rewards.append(round(acc_r/num_e, 2))
        total_drp_pkg_rate.append(round(sum(drp_pkg_rate.values())/len(drp_pkg_rate.values()), 2))
        print(f'act_frequency: {act_frequency}')
        print(f'mean episode rewards: {mean_episode_rewards[-1]}')
        print(f'mean episode length: {mean_episode_length[-1]}')
        print('mean dropped packages rate:', drp_pkg_rate)
        print('total mean dropped packages rate:', total_drp_pkg_rate[-1])
        print(f'Runtime: {end-start}S')

    print('Summary:')
    print('act_frequency', tuple([i for i in act_frequencies]))
    print('mean_episode_length', tuple(mean_episode_length))
    print('mean_episode_rewards', tuple(mean_episode_rewards))
    print('total_drp_pkg_rate', tuple(total_drp_pkg_rate))
