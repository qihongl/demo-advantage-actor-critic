import gym
import torch
import argparse
from models import A2C_discrete as Agent
from utils import run, get_env_info


def main(env_name, n_hidden):
    '''render the performance of a saved ckpt'''
    # define env and agent
    env = gym.make(env_name).env
    state_dim, n_actions, action_space_type = get_env_info(env)
    agent = Agent(state_dim, n_hidden, n_actions)
    agent.load_state_dict(torch.load(f'../log/agent-{env_name}.pth'))
    agent.eval()
    cumulative_reward, step, probs, rewards, values = run(
        agent, env, render=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', default='CartPole-v0')
    parser.add_argument('-n_hidden', default=128)
    args = parser.parse_args()
    print(f'Render an agent on {args.env}...')
    main(args.env, args.n_hidden)
