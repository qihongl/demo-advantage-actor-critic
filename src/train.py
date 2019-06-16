import gym
import torch
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from models import compute_returns, compute_a2c_loss
from models import A2C as Agent
from utils import run

sns.set(style='white', context='talk', palette='colorblind')


def main(env_name, n_epoch, learning_rate, gamma, seed_val=0):
    '''train an a2c network some gym env'''
    # define env and agent
    env = gym.make(env_name).env
    env.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = Agent(state_dim, n_actions)
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)

    # train
    log_steps = np.zeros((n_epoch,))
    log_loss_v = np.zeros((n_epoch,))
    log_loss_p = np.zeros((n_epoch,))
    for i in range(n_epoch):
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
            print('Epoch : %.3d | steps: %.2f' % (i, log_steps[i]))

    # save weights
    torch.save(agent.state_dict(), f'../log/agent-{env_name}.pth')

    '''show learning curve: return, steps'''
    f, ax = plt.subplots(1, 1, figsize=(5, 3), sharex=True)
    ax.plot(log_steps)
    ax.set_title('Learning curve')
    ax.set_ylabel('Return/#steps')
    sns.despine()
    f.tight_layout()
    f.savefig(f'../figs/lc-{env_name}.png', dpi=120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', default='CartPole-v0')
    parser.add_argument('-n_epoch', default=500, type=int)
    parser.add_argument('-lr', default=1e-3, type=float)
    parser.add_argument('-gamma', default=.99, type=float)
    parser.add_argument('-seed', default=0, type=int)
    args = parser.parse_args()
    print(f'Train an agent on {args.env}...')
    main(args.env, args.n_epoch, args.lr, args.gamma)
