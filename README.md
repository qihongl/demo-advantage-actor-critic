# demo-A2C

A demo of the advantage actor critic (A2C) algorithm (Mnih et al. 2016), tested on `cartpole-v0`. 

Here's the learned behavior: 

<img src="https://github.com/qihongl/demo-advantage-actor-critic/blob/master/figs/render.gif" width=500>

Here's the learning curve: 

<img src="https://github.com/qihongl/demo-advantage-actor-critic/blob/master/figs/lc.png" width=400>


### How to use: 

For training: 
```
python train-cartpole.py
```

For rendering the learned behavior:
```
python render-cartpole.py
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
    ├── render-cartpole.py       # visualize env 
    ├── train-cartpole.py        # train an agent
    └── utils.py

```

### Note: 

- I didn't tune the hyperparam, so the param setting is likely to be suboptimal. 
- This task doesn't seem to require discounting, but empirically having a `gamma` < 1 seem to be important.

### Reference: 

Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T. P., Harley, T., … Kavukcuoglu, K. (2016). Asynchronous Methods for Deep Reinforcement Learning. Retrieved from http://arxiv.org/abs/1602.01783

