from gym.envs.registration import register

register(
    id='rllib_network-v0',
    entry_point='custom_env.environment:RLlibEnv',
)

register(
    id='main_network-v0',
    entry_point='custom_env.environment:MainEnv',
)
