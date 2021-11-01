import json

import gym
from ray.tune.registry import register_env

from custom_env.environment import RLlibEnv
from dotdic import DotDic
from rllib_ppo import set_config
from ray.rllib.agents import ppo

if __name__ == '__main__':
    with open('./config/PartialAccess.json', 'r') as f:
        conf = json.loads(f.read())
    
    register_env(conf['env_name'], lambda _: RLlibEnv(DotDic(conf)))
    trainer = ppo.PPOTrainer(env=conf['env_name'])
    # config['env'] = [name]
    # config = set_config(conf)
