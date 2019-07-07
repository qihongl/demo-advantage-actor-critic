import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from models.utils import softmax, ortho_init, entropy


class A2C_discrete(nn.Module):
    """a A2C w/ multinomial action
    process: relu(Wx) -> pi, v

    Parameters
    ----------
    dim_input : int
        dim state space
    dim_hidden : int
        number of hidden units
    dim_output : int
        dim action space

    Attributes
    ----------
    i2h : torch.nn.Linear
        input to hidden mapping
    actor : torch.nn.Linear
        the actor network
    critic : torch.nn.Linear
        the critic network
    _init_weights : helper func
        default weight init scheme

    """

    def __init__(self, dim_input, dim_hidden, dim_output):
        super(A2C_discrete, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.i2h = nn.Linear(dim_input, dim_hidden)
        self.actor = nn.Linear(dim_hidden, dim_output)
        self.critic = nn.Linear(dim_hidden, 1)
        ortho_init(self)

    def forward(self, x, beta=1):
        """compute action distribution and value estimate, pi(a|s), v(s)

        Parameters
        ----------
        x : a vector
            a vector, state representation
        beta : float, >0
            softmax temp, big value -> more "randomness"

        Returns
        -------
        vector, scalar
            pi(a|s), v(s)

        """
        h = F.relu(self.i2h(x))
        v_t = self.critic(h)
        pi_a_t = softmax(self.actor(h), beta)
        a_t, log_prob_a_t = _sample_action(pi_a_t)
        ent_t = entropy(pi_a_t)
        # pack data
        misc = []
        output = [a_t, log_prob_a_t, ent_t, v_t, misc]
        return output


class A2C_linear(nn.Module):
    """a linear actor-critic network
    process: x -> pi, v

    Parameters
    ----------
    dim_input : int
        dim state space
    dim_output : int
        dim action space

    Attributes
    ----------
    actor : torch.nn.Linear
        the actor network
    critic : torch.nn.Linear
        the critic network

    """

    def __init__(self, dim_input, dim_output):
        super(A2C_linear, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.actor = nn.Linear(dim_input, dim_output)
        self.critic = nn.Linear(dim_input, 1)
        ortho_init(self)

    def forward(self, x, beta=1):
        """compute action distribution and value estimate, pi(a|s), v(s)

        Parameters
        ----------
        x : a vector
            a vector, state representation
        beta : float, >0
            softmax temp, big value -> more "randomness"

        Returns
        -------
        vector, scalar
            pi(a|s), v(s)

        """
        pi_a_t = softmax(self.actor(x), beta)
        v_t = self.critic(x)
        a_t, log_prob_a_t = _sample_action(pi_a_t)
        ent_t = entropy(a_t)
        # pack data
        misc = []
        output = [a_t, log_prob_a_t, ent_t, v_t, misc]
        return output


def _sample_action(pi_a_t):
    """action selection by sampling from a multinomial.

    Parameters
    ----------
    pi_a_t : 1d torch.tensor
        action distribution, pi(a|s)

    Returns
    -------
    torch.tensor(int), torch.tensor(float)
        sampled action, log_prob(sampled action)

    """
    m = Categorical(pi_a_t)
    a_t = m.sample()
    log_prob_a_t = m.log_prob(a_t)
    return a_t, log_prob_a_t
