import gym
import torch
from models import A2C as Agent
from utils import run


def main():
    '''render the performance of a saved ckpt'''
    # define env and agent
    env = gym.make('CartPole-v0').env
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = Agent(state_dim, n_actions)
    agent.load_state_dict(torch.load('../log/agent.pth'))
    agent.eval()
    step, probs, rewards, values = run(agent, env, render=True)


if __name__ == "__main__":
    main()
