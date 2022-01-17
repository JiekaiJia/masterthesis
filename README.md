# Queueing Network

Here is an instruction how to use RLlib train with our environment.

## Install

`pip install -r requirements.txt`

## Usage
***belief_train.py*** is used to train the belief model separately. ***model_components.py*** contains the
MVAE used as a belief communication model.

`python belief_train.py --use_belief`

***rllib_ppo.py*** is used to train the policy model with rllib default network structure. 
One should enter the experiment name. If he wants to train the policy which takes 
the belief as observations, then he should set `--use_belief`, otherwise, 
he could only enter the experiment name. `--test` is set for trained model test purpose, 
but man need to also copy the model parameter path into ***rllib_ppo.py***. Besides, man could 
change the model anf training setting in the file too.

`python rllib_ppo.py -experiment_name PPO_belief --use_belief`

***test_train.py*** is used to train the policy model with rllib but customizing 
loss function and network structure. The use of this file is similar to ***rllib_ppo.py***

`python test_train.py -experiment_name PPO_1e-6autoencoder1e-4_delhiddens --test`

***RLlib_custom_models.py*** is the file that contains the customised model. The models are all end-to-end models 
that can directly be trained with rllib. ***custom_PPO.py*** is used to customise the loss function. Because we change 
the network structure, it's important to use a proper loss function instead of the default one to help train the model 
efficiently.


