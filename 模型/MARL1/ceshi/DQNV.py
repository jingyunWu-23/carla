import gym

from stable_baselines3 import DQN


import numpy as np
# Create environment


# load model
model = DQN.load("highway_dqn_model")
states=None
list=[[[ 1.   ,       0.9940078  , 0.      ,    0.41666666 , 0.        ],  [ 1.   ,       0.16675936 , 0.       ,  -0.0463527  , 0.        ],  [ 1.  ,        0.338202 ,   0.     ,    -0.0175555  , 0.        ],  [ 1.  ,        0.49176827 , 0.33333334, -0.04948474,  0.        ],  [ 1.   ,       0.6366527 ,  0.33333334, -0.03372754 , 0.        ]]]
observations=np.array(list)
actions = model.predict(observations)

print(actions)