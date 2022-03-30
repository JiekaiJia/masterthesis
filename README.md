# Decentralised Coordination in Partially Observable Queueing Networks

This is my master thesis, using reinforcement learning to control packets dispatch to minimize the 
packet drop rate. Due to the partial observation of the agents, we implement a communication channel 
between agents to improve the agents sensing to the environment.

## Install

`pip install -r requirements.txt`

## Usage
***main.py*** is the core interface for experiments. We can use CLI to control the environment and neural network setting.
`-experiment_name` is used to name the experiment. `--silent` is a gate controlling communication between agents. 
`--default` indicates whether to use a 2-layer MLP or customized model for agents. `--JSQ` controls whether we use a heuristic
policy "join shortest queue" for agents. If we want to test a tarined model, writing down `--test` works. 
`--true_obs` and `--opposite` control the environment observations. If setting `--true_obs`, the agents get true observation,
otherwise the agents get opposite observations. If setting neither of them, the agents get delayed observations. Some example
CLIs is listed here:

Test JSQ policy with delayed observations(using `--true_obs` or `--opposite` change the observation type).

`python main.py -experiment_name JSQ --default --test --JSQ`

Train 2-layer MLP with true observations(the agents have no communication even without setting `--silent`)

`python main.py -experiment_name PPO_3agents --default --true_obs`

Train CommNet with opposite observations(changing experiment name to BiCNetxxx or ATCVxxx to use other communication models).

`python main.py -experiment_name CommNet_3agents_opposite --opposite`

***custom_models.py*** is the file that contains ATVC, CommNet, BiCNet. The models are all end-to-end trainable models. 

***custom_loss.py***, here, we implement the ATVC loss, which is an extension of PPO loss. Because we 
use VAE in ATVC, a VAE loss (ELBO) is then added to the original loss function.

***environment.py*** contains the environment we use in experiments. It's a queueing network with M
schedulers and S servers. the schedulers (agents) are controlled by neural network and observe the state
of servers, further they diapatch packets based on their own observations.

***PartialAccess.json*** controls the queueing network. We can define, for example, the number of agents, packet arrival
rate etc.


