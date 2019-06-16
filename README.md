# demo-A2C

A demo of the advantage actor critic (A2C) algorithm, tested on `cartpole-v0`. Here's the learned behavior: 

<img src="https://github.com/qihongl/demo-advantage-actor-critic/blob/master/figs/render.gif" width=500>

Here's the learning curve: 

<img src="https://github.com/qihongl/demo-advantage-actor-critic/blob/master/figs/lc.png" width=400>


### Note: 

- This task doesn't seem to require discounting, but empirically having a `gamma` < 1 seem to be important.

### Reference: 

Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T. P., Harley, T., … Kavukcuoglu, K. (2016). Asynchronous Methods for Deep Reinforcement Learning. Retrieved from http://arxiv.org/abs/1602.01783

