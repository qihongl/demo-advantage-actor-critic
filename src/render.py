import gym
import torch
import argparse

from models import A2C as Agent
from utils import run


def main(env_name):
    '''render the performance of a saved ckpt'''
    # define env and agent
    env = gym.make(env_name).env
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = Agent(state_dim, n_actions)
    agent.load_state_dict(torch.load(f'../log/agent-{env_name}.pth'))
    agent.eval()
    cumulative_reward, step, probs, rewards, values = run(
        agent, env, render=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', default='CartPole-v0')
    args = parser.parse_args()
    print(f'Render an agent on {args.env}...')
    main(args.env)
