import gym
import torch
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from models import compute_returns, compute_a2c_loss
from models import A2C as Agent
from utils import run
plt.switch_backend('agg')
sns.set(style='white', context='talk', palette='colorblind')


def main(
        env_name, n_epoch, learning_rate, gamma, n_hidden,
        seed_val=0, max_steps=1000
):
    '''train an a2c network some gym env'''
    # define env
    env = gym.make(env_name).env
    env.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    # define agent
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = Agent(state_dim, n_actions, dim_hidden=n_hidden)
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 'max', patience=100, factor=1/2, verbose=True)
    # train
    log_return = np.zeros((n_epoch,))
    log_step = np.zeros((n_epoch,))
    log_loss_v = np.zeros((n_epoch,))
    log_loss_p = np.zeros((n_epoch,))
    for i in range(n_epoch):
        cumulative_reward, step, probs, rewards, values = run(
            agent, env, gamma=gamma, max_steps=max_steps
        )
        # update weights
        returns = compute_returns(rewards, gamma=gamma, normalize=True)
        loss_policy, loss_value = compute_a2c_loss(probs, values, returns)
        loss = loss_policy + loss_value
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step(cumulative_reward)
        # log message
        log_return[i] = cumulative_reward
        log_step[i] = step
        log_loss_v[i] = loss_value.item()
        log_loss_p[i] = loss_policy.item()
        if np.mod(i, 10) == 0:
            print(
                'Epoch : %.3d | R: %.2f, steps: %4d | L: pi: %.2f, V: %.2f' %
                (i, log_return[i], log_step[i], log_loss_p[i], log_loss_v[i])
            )

    # save weights
    torch.save(agent.state_dict(), f'../log/agent-{env_name}.pth')

    '''show learning curve: return, steps'''
    f, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    axes[0].plot(log_return)
    axes[1].plot(log_step)
    axes[0].set_title(f'Learning curve: {env_name}')
    axes[0].set_ylabel('Return')
    axes[1].set_ylabel('#steps')
    axes[1].set_xlabel('Epoch')
    sns.despine()
    f.tight_layout()
    f.savefig(f'../figs/lc-{env_name}.png', dpi=120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', default='CartPole-v0')
    parser.add_argument('-n_epoch', default=500, type=int)
    parser.add_argument('-n_hidden', default=128, type=int)
    parser.add_argument('-lr', default=1e-3, type=float)
    parser.add_argument('-gamma', default=.99, type=float)
    parser.add_argument('-max_steps', default=2000, type=int)
    parser.add_argument('-seed', default=0, type=int)
    args = parser.parse_args()
    print(f'Train an agent on {args.env}...')
    main(
        args.env, args.n_epoch, args.lr, args.gamma, args.n_hidden,
        seed_val=args.seed, max_steps=args.max_steps
    )
