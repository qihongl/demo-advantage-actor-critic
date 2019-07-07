import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

eps = np.finfo(np.float32).eps.item()


class A2C_continuous(nn.Module):
    """a A2C w/ gaussian action

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
    h_mu : torch.nn.Linear
        hidden to mean mapping
    h_sd : torch.nn.Linear
        hidden to sd mapping
    critic : torch.nn.Linear
        the critic network
    dim_input
    dim_hidden
    dim_output

    """

    def __init__(self, dim_input, dim_hidden, dim_output):
        super(A2C_continuous, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.i2h = nn.Linear(dim_input, dim_hidden)
        self.h_mu = nn.Linear(dim_hidden, dim_output)
        self.h_sd = nn.Linear(dim_hidden, dim_output)
        self.critic = nn.Linear(dim_hidden, 1)

    def forward(self, x):
        h_t = self.i2h(x)
        v_t = self.critic(h_t)
        mu_t = self.h_mu(h_t)
        sd_t = self.h_sd(h_t)
        a_t, log_prob_a_t, ent_t = _choose_action(mu_t, sd_t)
        # pack data
        misc = [mu_t, sd_t]
        output = [a_t, log_prob_a_t, ent_t, v_t, misc]
        return output


def _choose_action(loc, scale):
    """sample an action from gaussian distribution, given its parameters

    Parameters
    ----------
    loc : torch.tensor
        the mean parameter
    scale : torch.tensor
        the scale parameter

    Returns
    -------
    type
        Description of returned object.

    """
    m = Normal(loc, scale)
    a_t = m.sample()
    log_prob_a_t = m.log_prob(a_t)
    ent_t = gaussian_entropy(m)
    return a_t, log_prob_a_t, ent_t


def gaussian_entropy(m_):
    """
    https://en.wikipedia.org/wiki/Normal_distribution#Maximum_entropy
    """
    return .5 + .5 * (np.log(2 * np.pi) + torch.log(m_.scale))
