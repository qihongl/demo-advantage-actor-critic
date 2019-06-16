# demo-A2C

A demo of the advantage actor critic (A2C) algorithm (Mnih et al. 2016), tested on `cartpole-v0`. 

Here's the learned behavior: 

<img src="https://github.com/qihongl/demo-advantage-actor-critic/blob/master/figs/render-CartPole-v0.gif" width=500>

Here's the learning curve: 

<img src="https://github.com/qihongl/demo-advantage-actor-critic/blob/master/figs/lc-CartPole-v0.png" width=400>


### How to use: 

For training: 
```
python train.py
```

For rendering the learned behavior:
```
python render.py
```

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

### Note: 

- I didn't tune hyperparameters, so the parameter setting is likely to be suboptimal. 
- Empirically, having a `gamma` close to 1 seem to be important.

### Reference: 

Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T. P., Harley, T., … Kavukcuoglu, K. (2016). Asynchronous Methods for Deep Reinforcement Learning. Retrieved from http://arxiv.org/abs/1602.01783

