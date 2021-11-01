from copy import deepcopy

from ray.rllib.agents.registry import get_trainer_class
from ray.tune.registry import register_env
import torch

from custom_env.environment import RLlibEnv
from dotdic import DotDic


def set_config(conf):
    conf['belief_training'] = False
    # Create test environment.
    test_env = RLlibEnv(DotDic(conf))
    # Register env
    env_name = conf['env_name']
    register_env(env_name, lambda _: RLlibEnv(DotDic(conf)))

    # The used algorithm
    alg_name = conf['alg_name']
    # Gets default training configuration.
    config = deepcopy(get_trainer_class(alg_name)._default_config)

    # === Settings for Rollout Worker processes ===
    # Use GPUs if `RLLIB_NUM_GPUS` env var set to > 0.
    # config['num_gpus'] = 0.0001
    # int(os.environ.get('RLLIB_NUM_GPUS', '0'))
    # config['num_gpus_per_worker'] = (1-0.0001)/3
    # Number of rollout worker actors to create for parallel sampling.
    config['num_workers'] = 3  # euler 20
    # config['num_envs_per_worker'] = 1

    # === Settings for the Trainer process ===
    # Whether layers should be shared for the value function.
    config['model'] = {
        'fcnet_hiddens': [128],
        'fcnet_activation': 'relu',
        # 'vf_share_layers': False,
        # 'use_lstm': True,
        # 'max_seq_len': 40,
        # 'lstm_use_prev_action': True,
        # 'lstm_use_prev_reward': True,
    }

    # === Environment Settings ===
    config['env'] = env_name
    # the env_creator function via the register env lambda below.
    # config['env_config'] = {'max_cycles': max_cycles, 'num_agents': num_agents, 'local_ratio': local_ratio}

    # # === Debug Settings ===
    # # Periodically print out summaries of relevant internal dataflow(DEBUG, INFO, WARN, or ERROR.)
    config['log_level'] = 'WARN'
    config['no_done_at_end'] = True

    # === Settings for Multi-Agent Environments ===
    # Configuration for multi-agent setup with policy sharing:
    config['multiagent'] = {
        # Map of type MultiAgentPolicyConfigDict from policy ids to tuples
        # of (policy_cls, obs_space, act_space, config). This defines the
        # observation and action spaces of the policies and any extra config.
        'policies': {
            'shared': (None, test_env.observation_spaces[agent], test_env.action_spaces[agent], {}) for agent in test_env.schedulers
        },
        # Function mapping agent ids to policy ids.
        'policy_mapping_fn': lambda agent_id: 'shared',
    }
    return config


if __name__ == '__main__':
    import argparse
    import json

    import ray

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default=None,
                        help="gives the belief model's name [default: None]")
    parser.add_argument('--silent', action='store_true', default=False,
                        help='defines if scheduler can communicate [default: False]')
    parser.add_argument('--use_belief', action='store_true', default=False,
                        help='encodes observations to belief [default: False]')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training [default: False]')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    with open('./config/PartialAccess.json', 'r') as f:
        conf = json.loads(f.read())
    
    conf['use_belief'] = args.use_belief
    conf['silent'] = args.silent
    if args.use_belief:
        assert args.model_name is not None, 'If use belief model, the model name must be given.'
        conf['model_name'] = args.model_name
  
    config = set_config(conf)
    # Initialize ray and trainer object
    ray.init(
        ignore_reinit_error=True,
        local_mode=True,
        # log_to_driver=False
    )

    # Stop criteria
    stop = {
        "training_iteration": conf['training_iteration'],
    }
    
    # if conf['restore_from_previous_model']:
    # Train
    results = ray.tune.run(
        conf['alg_name'],
        stop=stop,
        config=config,
        checkpoint_at_end=True,
        local_dir=conf['local_dir'],
        checkpoint_freq=conf['checkpoint_freq'],
        restore='/content/drive/MyDrive/Data Science/pythonProject/masterthesis/ray_results/PPO/PPO_rllib_network-v0_c57d5_00000_0_2021-10-31_13-09-25/checkpoint_000050/checkpoint-50'
    )

    ray.shutdown()
