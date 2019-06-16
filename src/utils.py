import torch


def run(agent_, env, render=False, max_steps=1000):
    s_t = env.reset()
    probs, rewards, values = [], [], []
    step = 0
    while step < max_steps:
        if render:
            env.render()
        pi_a_t, v_t = agent_.forward(to_th(s_t).view(1, -1))
        a_t, prob_a_t = agent_.pick_action(pi_a_t)
        s_t, r_t, done, info = env.step(int(a_t))
        probs.append(prob_a_t)
        rewards.append(r_t)
        values.append(v_t)
        step += 1
        if done:
            break
    env.close()
    return step, probs, rewards, values


def to_th(np_array):
    return torch.tensor(np_array).type(torch.FloatTensor)
