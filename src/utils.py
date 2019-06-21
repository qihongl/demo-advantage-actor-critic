import numpy as np
import torch
# import time


def run(agent_, env, gamma=.99, render=False, max_steps=1000):
    s_t = env.reset()
    probs, rewards, values = [], [], []
    step = 0
    cumulative_reward = 0
    while step < max_steps:
        if render:
            env.render()
            # time.sleep(.07)
        pi_a_t, v_t = agent_.forward(to_th(s_t).view(1, -1))
        a_t, prob_a_t = agent_.pick_action(pi_a_t)
        s_t, r_t, done, info = env.step(int(a_t))
        probs.append(prob_a_t)
        rewards.append(r_t)
        values.append(v_t)
        cumulative_reward += r_t * gamma ** step
        step += 1
        if done:
            break
    env.close()
    return cumulative_reward, step, probs, rewards, values


def get_state_dim(env):
    state_shape = env.observation_space.shape
    if len(state_shape) == 0:
        ns = 1
    else:
        ns = np.cumprod([i for i in state_shape])[-1]
    return ns


def to_th(np_array):
    return torch.tensor(np_array).type(torch.FloatTensor)
