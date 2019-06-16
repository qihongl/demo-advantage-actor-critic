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
np.random.seed(0)
torch.manual_seed(0)
img_dir = '../figs'


def main():
    '''train an a2c network on cartpole-v0'''
    # define env and agent
    env = gym.make('CartPole-v0').env
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # training params
    n_epochs = 500
    dim_hidden = 128
    dropout_rate = .5
    learning_rate = 1e-3
    gamma = .99

    agent = Agent(
        dim_input=state_dim, dim_hidden=dim_hidden, dim_output=n_actions,
        dropout_rate=dropout_rate
    )
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)

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

    f, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    axes[0].plot(log_steps)
    axes[0].set_title('Learning curve')
    axes[0].set_ylabel('Return')

    axes[1].plot(log_loss_v)
    axes[1].set_title(' ')
    axes[1].set_ylabel(r'Loss, $V_t$')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylim([0, None])
    sns.despine()
    f.tight_layout()
    f.savefig(os.path.join(img_dir, 'lc.png'), dpi=120)

    # step, probs, rewards, values = run(agent, True)


if __name__ == "__main__":
    main()
