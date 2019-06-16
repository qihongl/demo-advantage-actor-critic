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
    dim_hidden = 128
    dropout_rate = .5
    agent = Agent(
        dim_input=state_dim, dim_hidden=dim_hidden, dim_output=n_actions,
        dropout_rate=dropout_rate
    )
    agent.load_state_dict(torch.load('../log/agent.pth'))
    agent.eval()
    step, probs, rewards, values = run(agent, env, render=True)


if __name__ == "__main__":
    main()
