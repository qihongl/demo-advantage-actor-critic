# demo-A2C

A demo of the advantage actor critic (A2C) algorithm (Mnih et al. 2016). 

Here's its learned behavior on `CartPole-v0`. The goal is to keep the pole upright. 

<img src="https://github.com/qihongl/demo-advantage-actor-critic/blob/master/figs/render-CartPole-v0.gif" width=500>

Here's the learning curve: 

<img src="https://github.com/qihongl/demo-advantage-actor-critic/blob/master/figs/lc-CartPole-v0.png" width=450>


### How to use: 

The dependencies are: `pytorch`, `gym`, `numpy`, `matplotlib`, `seaborn`. The lastest version should work. 

For training (the default environment is `CartPole-v0`): 
```
python train.py
```

For rendering the learned behavior:
```
python render.py
```

The agent should be runnable on any environemnt with a discrete action space. To run the agent on some other environment, type `python train.py -env ENVIRONMENT_NAME`.

For example, the same architecture can also solve `Acrobot-v1`: 

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

[1] 
Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T. P., Harley, T., … Kavukcuoglu, K. (2016). Asynchronous Methods for Deep Reinforcement Learning. Retrieved from http://arxiv.org/abs/1602.01783

[2] 
Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., & Zaremba, W. (2016). OpenAI Gym. Retrieved from http://arxiv.org/abs/1606.01540

[3] 
Paszke, A., Gross, S., Chintala, S., Chanan, G., Yang, E., DeVito, Z., … Lerer, A. (2017). Automatic differentiation in PyTorch. Retrieved from https://openreview.net/forum?id=BJJsrmfCZ

[4] 
<a href="https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py">pytorch/examples/reinforcement_learning/actor_critic</a>

[5] 
<a href="http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-6.pdf">Slides</a> from 
Deep Reinforcement Learning, CS294-112 at UC Berkeley
