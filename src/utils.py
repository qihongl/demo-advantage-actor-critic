import numpy as np
import torch
import time
import pdb
time_tick = 0


def run(agent, env, gamma=.99, render=False, max_steps=1000):
    s_t = env.reset()
    probs, rewards, values = [], [], []
    step = 0
    cumulative_reward = 0
    while step < max_steps:
        if render:
            env.render()
            time.sleep(time_tick)
        # forward prop
        [a_t, log_prob_a_t, ent_t, v_t, misc] = agent.forward(
            to_th(s_t).view(1, -1))
        # make transition
        s_t, r_t, done, info = env.step(to_np(a_t))
        # cache info
        probs.append(log_prob_a_t)
        rewards.append(r_t)
        values.append(v_t)
        cumulative_reward += r_t * gamma ** step
        step += 1
        if done:
            break
    env.close()
    return cumulative_reward, step, probs, rewards, values


def get_env_info(env):
    state_dim = get_state_dim(env)
    n_actions, action_space_type = get_action_space_type(env)
    return state_dim, n_actions, action_space_type


def get_state_dim(env):
    state_shape = env.observation_space.shape
    if len(state_shape) == 0:
        ns = 1
    else:
        ns = np.cumprod([i for i in state_shape])[-1]
    return ns


def get_action_space_type(env):
    action_space_str = str(env.action_space)
    if 'Box' in action_space_str:
        action_space_type = 'continuous'
        n_actions = env.action_space.shape[0]
    elif 'Discrete' in action_space_str:
        action_space_type = 'discrete'
        n_actions = env.action_space.n
    else:
        error_message = f'Unrecognizable action space type {action_space_str}'
        raise ValueError(error_message)
    return n_actions, action_space_type


def to_th(np_array):
    return torch.tensor(np_array).type(torch.FloatTensor)


def to_np(th_tensor):
    return th_tensor.data.numpy()
