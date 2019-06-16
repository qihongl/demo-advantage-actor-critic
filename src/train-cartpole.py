import os
import gym
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utils import run
from models import A2C as Agent
from models import compute_returns, compute_a2c_loss

sns.set(style='white', context='talk', palette='colorblind')
seed_val = 0
np.random.seed(seed_val)
torch.manual_seed(seed_val)
img_dir = '../figs'


def main():
    '''train an a2c network on cartpole-v0'''
    # define env and agent
    env = gym.make('CartPole-v0').env
    env.seed(seed_val)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # training params
    n_epochs = 500
    learning_rate = 1e-3
    gamma = .99
    agent = Agent(state_dim, n_actions)
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)

    # train
    log_steps = np.zeros((n_epochs,))
    log_loss_v = np.zeros((n_epochs,))
    log_loss_p = np.zeros((n_epochs,))
    for i in range(n_epochs):
        step, probs, rewards, values = run(agent, env)
        # update weights
        returns = compute_returns(rewards, gamma=gamma)
        loss_policy, loss_value = compute_a2c_loss(probs, values, returns)
        loss = loss_policy + loss_value
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log message
        log_steps[i] = step
        log_loss_v[i] = loss_value.item()
        log_loss_p[i] = loss_policy.item()
        if np.mod(i, 50) == 0:
            print('Epoch : %.3d | return: %.2f | loss: v: %.2f, p: %.2f' % (
                i, log_steps[i], log_loss_v[i], log_loss_p[i]))

    torch.save(agent.state_dict(), '../log/agent.pth')

    '''show learning curve: return, steps'''
    f, ax = plt.subplots(1, 1, figsize=(5, 3), sharex=True)
    ax.plot(log_steps)
    ax.set_title('Learning curve')
    ax.set_ylabel('Return (#steps)')
    sns.despine()
    f.tight_layout()
    f.savefig(os.path.join(img_dir, 'lc.png'), dpi=120)


if __name__ == "__main__":
    main()
