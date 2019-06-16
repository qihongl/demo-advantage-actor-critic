import torch.nn as nn
import torch.nn.functional as F

from models.utils import softmax, ortho_init


class A2C(nn.Module):
    """a MLP actor-critic network
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
    ih : torch.nn.Linear
        input to hidden mapping
    actor : torch.nn.Linear
        the actor network
    critic : torch.nn.Linear
        the critic network
    _init_weights : helper func
        default weight init scheme

    """

    def __init__(self, dim_input, dim_hidden, dim_output, dropout_rate=0):
        super(A2C, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.ih = nn.Linear(dim_input, dim_hidden)
        self.actor = nn.Linear(dim_hidden, dim_output)
        self.critic = nn.Linear(dim_hidden, 1)
        self.dropout = nn.Dropout(p=dropout_rate)
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
        h = F.relu(self.ih(x))
        h = self.dropout(h)
        action_distribution = softmax(self.actor(h), beta)
        value_estimate = self.critic(h)
        return action_distribution, value_estimate


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
        action_distribution = softmax(self.actor(x), beta)
        value_estimate = self.critic(x)
        return action_distribution, value_estimate
