import torch
import torch.nn as nn
import numpy as np

from VisualRL.rllib.common.distributions import SquashedDiagGaussianDistribution
from VisualRL.rllib.common.utils import weight_init

LOGSTD_MIN = -20
LOGSTD_MAX = 2
class Actor(nn.Module):
    def __init__(
            self,
            action_space,
            feature_extractor,
            feature_dims,
            ):
        super(Actor, self).__init__()

        self.action_space = action_space
        self.feature_extractor = feature_extractor
        self.feature_dims = feature_dims
        self.optimizer = None
        self.log_std_init = 0.

        self.latent_pi_net = nn.Sequential(
                nn.Linear(feature_dims, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 64),nn.ReLU(),
                )
        self.action_dist = SquashedDiagGaussianDistribution(action_space)
        self.mu = nn.Linear(64, action_space)
        self.log_std = nn.Linear(64, action_space)

        self.apply(weight_init)

    def get_action_parameters(self, obs):
        features = self.feature_extractor(obs)
        latent_pi = self.latent_pi_net(features)
        mean_actions = self.mu(latent_pi)
        log_std = torch.clamp(self.log_std(latent_pi), LOGSTD_MIN, LOGSTD_MAX)

        return mean_actions, log_std

    def forward(self, obs, determinstic = False):
        mean_actions, log_std = self.get_action_parameters(obs)

        return self.action_dist.actions_from_params(mean_actions, log_std, determinstic = determinstic)

    def action_log_prob(self, obs):
        mean_actions, log_std = self.get_action_parameters(obs)

        return self.action_dist.log_prob_from_params(mean_actions, log_std)
