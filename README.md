# demo-A2C

A demo of the advantage actor critic (A2C) algorithm (Mnih et al. 2016). 

Here's its learned behavior on `cartpole-v0`. The goal is to keep the pole upright. 

<img src="https://github.com/qihongl/demo-advantage-actor-critic/blob/master/figs/render-CartPole-v0.gif" width=500>

Here's the learning curve: 

<img src="https://github.com/qihongl/demo-advantage-actor-critic/blob/master/figs/lc-CartPole-v0.png" width=450>


### How to use: 

The dependencies are: `pytorch`, `gym`, `numpy`, `matplotlib`, `seaborn`. The lastest version should work. 

For training (the default environment is `cartpole-v0`): 
```
python train.py
```

For rendering the learned behavior:
```
python render.py
```

The agent should be runnable on any environemnt with a discrete action space. To run the agent on other environment, type `python train.py -env ENVIRONMENT_NAME`. For other input arg options, see the `main` function in `src/train.py`.

For example, the same architecture can also solve `acrobot-v1`: 

<img src="https://github.com/qihongl/demo-advantage-actor-critic/blob/master/figs/render-Acrobot-v1.gif" width=400>


... and `LunarLander-v2`: 

<img src="https://github.com/qihongl/demo-advantage-actor-critic/blob/master/figs/render-LunarLander-v2.gif" width=400>



### dir structure: 
```
.
├── LICENSE
├── README.md
├── figs                         # figure dir
├── log                          # logging dir 
│   └── agent.pth                # a pre-trained agent
└── src
    ├── models
    │   ├── A2C.py               # a2c
    │   ├── __init__.py
    │   ├── _a2c_helpers.py      # a2c helper 
    │   └── utils.py
    ├── render.py                # visualize env 
    ├── train.py                 # train an agent
    └── utils.py

```

### Reference: 

Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T. P., Harley, T., … Kavukcuoglu, K. (2016). Asynchronous Methods for Deep Reinforcement Learning. Retrieved from http://arxiv.org/abs/1602.01783

Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., & Zaremba, W. (2016). OpenAI Gym. Retrieved from http://arxiv.org/abs/1606.01540
