import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli, Binomial

import numpy as np


def sum_independent_dims(tensor):
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor

class DiagGaussianDistribution():
    def __init__(self, action_dim):
        self.distribution = None
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution(self, mean_actions, log_std):
        action_std = torch.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std, validate_args=False)
        return self

    def log_prob(self, actions):
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self):
        return sum_independent_dims(self.distribution.entropy())

    def sample(self):
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self):
        return self.distribution.mean

    def get_actions(self, determinstic=False):
        if determinstic:
            return self.mode()
        return self.sample()

    def actions_from_params(self, mean_actions, log_std, determinstic=False):
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(determinstic=determinstic)

    def log_prob_from_params(self, mean_actions, log_std):
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class SquashedDiagGaussianDistribution(DiagGaussianDistribution):
    """
    Gaussian distribution with diagonal covariance matrix, followed by a squashing function (tanh) to ensure bounds.

    :param action_dim: Dimension of the action space.
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, action_dim: int, epsilon = 1e-6):
        super(SquashedDiagGaussianDistribution, self).__init__(action_dim)
        # Avoid NaN (prevents division by zero or log of zero)
        self.epsilon = epsilon
        self.gaussian_actions = None

    def proba_distribution(self, mean_actions, log_std):
        super(SquashedDiagGaussianDistribution, self).proba_distribution(mean_actions, log_std)
        return self

    def log_prob(self, actions, gaussian_actions):
        # Inverse tanh
        # Naive implementation (not stable): 0.5 * torch.log((1 + x) / (1 - x))
        # We use numpy to avoid numerical instability
        if gaussian_actions is None:
            # It will be clipped to avoid NaN when inversing tanh
            gaussian_actions = TanhBijector.inverse(actions)

        # Log likelihood for a Gaussian distribution
        log_prob = super(SquashedDiagGaussianDistribution, self).log_prob(gaussian_actions)
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        log_prob -= torch.sum(torch.log(1 - actions ** 2 + self.epsilon), dim=1)
        return log_prob

    def entropy(self):
        # No analytical form,
        # entropy needs to be estimated using -log_prob.mean()
        raise ValueError("trying to compute entropy for squash gaussian distributions")
        return None

    def sample(self):
        # Reparametrization trick to pass gradients
        self.gaussian_actions = super().sample()
        return torch.tanh(self.gaussian_actions)

    def mode(self):
        self.gaussian_actions = super().mode()
        # Squash the output
        return torch.tanh(self.gaussian_actions)

    def log_prob_from_params(self, mean_actions, log_std):
        action = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(action, self.gaussian_actions)
        return action, log_prob

class TanhBijector:
    """
    Bijective transformation of a probability distribution
    using a squashing function (tanh)
    TODO: use Pyro instead (https://pyro.ai/)

    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, epsilon = 1e-6):
        self.epsilon = epsilon

    @staticmethod
    def forward(x):
        return torch.tanh(x)

    @staticmethod
    def atanh(x):
        """
        Inverse of Tanh

        Taken from Pyro: https://github.com/pyro-ppl/pyro
        0.5 * torch.log((1 + x ) / (1 - x))
        """
        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(y):
        """
        Inverse tanh.

        :param y:
        :return:
        """
        eps = torch.finfo(y.dtype).eps
        # Clip the action to avoid NaN
        return TanhBijector.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))

    def log_prob_correction(self, x):
        # Squash correction (from original SAC implementation)
        return torch.log(1.0 - torch.tanh(x) ** 2 + self.epsilon)
