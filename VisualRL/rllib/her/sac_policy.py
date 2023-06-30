import torch
import torch.nn as nn

import numpy as np

from VisualRL.rllib.her.actor import Actor
from VisualRL.rllib.her.critic import Critic
from VisualRL.rllib.common.torch_layers import make_feature_extractor

class SACPolicy(nn.Module):
    def __init__(
            self,
            observation_space,
            action_space,
            feature_dims,
            feature_extractor,
            rollout_buffer,
            device,
            min_action,
            max_action,
            learning_rate,
            net_class,
            ):
        super(SACPolicy, self).__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.feature_dims = feature_dims
        self.feature_extractor = feature_extractor
        self.rollout_buffer = rollout_buffer
        self.device = device
        self.min_action = min_action
        self.max_action = max_action
        self.initial_learning_rate = learning_rate
        self.net_class = net_class

        # entropy item
        self.target_entropy = -np.prod(self.action_space).astype(np.float32)
        self.log_ent_coef = torch.log(torch.ones(1, device = self.device)).requires_grad_(True)
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr = 1e-3)
        self.ent_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.ent_coef_optimizer, 0.999)
        # build actor and critic
        self.actor = self.make_actor()
        self.actor.optimizer = torch.optim.Adam(self.actor.parameters(), lr = self.initial_learning_rate)
        self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.actor.optimizer, 0.999)

        self.critic, self.critic_parameters = self.make_critic(feature_extractor = self.feature_extractor)
        self.critic.optimizer = torch.optim.Adam(self.critic_parameters, lr = self.initial_learning_rate)
        self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.critic.optimizer, 0.999)

        self.critic_target, _ = self.make_critic()
        self.critic_target.load_state_dict(self.critic.state_dict())

    def make_actor(self):
        actor_kwargs = dict()
        actor_kwargs["action_space"] = self.action_space
        actor_kwargs["feature_extractor"] = self.feature_extractor
        actor_kwargs["feature_dims"] = self.feature_dims

        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, feature_extractor = None):
        critic_kwargs = dict()
        if feature_extractor is None:
            critic_kwargs["feature_extractor"] = make_feature_extractor(self.net_class, self.observation_space, self.feature_dims, self.device)
        else:
            critic_kwargs["feature_extractor"] = self.feature_extractor

        critic_kwargs["observation_space"] = self.observation_space
        critic_kwargs["action_space"] = self.action_space
        critic_kwargs["feature_dims"] = self.feature_dims

        critic = Critic(**critic_kwargs).to(self.device)
        critic_parameters = critic.get_parameters()

        return critic, critic_parameters

    def predict(self, observations, determinstic):
        return self.actor(observations, determinstic)
