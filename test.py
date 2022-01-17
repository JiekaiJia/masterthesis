"""Test file for JSQ and random policy."""

import json
import time
import numpy as np
import scipy.stats as st
from scipy.special import softmax
import tqdm

from custom_env.environment import MainEnv
from logger import get_logger
import policies
from utils import DotDic


logger = get_logger(__name__, f'policy_test_{str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))}.log')


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
    std_episode_rewards, std_episode_length, std_drp_pkg_rate = [], [], []
    # act_frequencies = args.act_frequency
    act_frequencies = range(6)
    # act_frequencies = [0]
    for act_frequency in act_frequencies:
        episode_rewards, episode_length, episode_drp_pkg_rate = [], [], []
        config.act_frequency = act_frequency
        # Initialize Environment
        env = MainEnv(config)
        # Initialize Policy
        policy = policies.JSQPolicy(config, env)
        # intention map
        intention_map = np.zeros((6, 6))
        counter = np.zeros((6, 6))

        logger.info(f'{policy.__class__.__name__} start running!!')
        num_e = 120
        for _ in tqdm.tqdm(range(num_e)):
            obss = env.reset()
            real_obss = obss
            acc_r, step = 0, 0
            dones = {'__all__': False}
            drop_pkg = {scheduler: 0 for scheduler in env.schedulers}
            while not dones['__all__']:
                # print('timestep:', step + 1)
                actions = {scheduler: policy.compute_actions(obs) for scheduler, obs in real_obss.items()}
                if "scheduler_1" in real_obss:
                    o = real_obss["scheduler_1"]
                    intention_map[o[0], o[1]] += softmax(actions["scheduler_1"])[0]
                    counter[o[0], o[1]] += 1

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
            for k, v in env.acc_drop_pkgs.items():
                drop_pkg[k] += v
            episode_length.append(step)
            episode_rewards.append(acc_r)
            episode_drp_pkg_rate.append(sum(drop_pkg.values())/(len(drop_pkg)*config.n_packages))
        end = time.time()
        mean_r = np.mean(episode_rewards)
        mean_p = np.mean(episode_drp_pkg_rate)
        mean_episode_rewards.append(mean_r)
        std_episode_rewards.append(st.t.interval(0.95, len(episode_rewards)-1, loc=mean_r, scale=st.sem(episode_rewards)))
        total_drp_pkg_rate.append(mean_p)
        std_drp_pkg_rate.append(st.t.interval(0.95, len(episode_drp_pkg_rate)-1, loc=mean_p, scale=st.sem(episode_drp_pkg_rate)))
        print(f'act_frequency: {act_frequency}')
        print(f'mean episode rewards: {mean_episode_rewards[-1]}, std episode rewards: {std_episode_rewards[-1]}')
        print(f'total mean dropped packages rate: {total_drp_pkg_rate[-1]}, std_drp_pkg_rate: {std_drp_pkg_rate[-1]}')
        print(f'Runtime: {end-start}S')

    intention_map /= (counter+1e-8)
    print('Summary:')
    logger.info(f'act_frequency{tuple([i for i in act_frequencies])}')
    logger.info(f'mean_episode_rewards: {tuple(mean_episode_rewards)}, std episode rewards: {tuple(std_episode_rewards)}')
    logger.info(f'total_drp_pkg_rate: {tuple(total_drp_pkg_rate)}, std_drp_pkg_rate: {tuple(std_drp_pkg_rate)}')
    logger.info(f'intention map:{intention_map}')
    logger.info(f'counts for each combination {counter}')
