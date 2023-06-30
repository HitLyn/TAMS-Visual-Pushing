import torch
import torch.nn as nn
import numpy as np
from IPython import embed
from VisualRL.rllib.common.utils import weight_init

class Critic(nn.Module):
    def __init__(
            self,
            observation_space,
            action_space,
            feature_extractor,
            feature_dims,
            ):
        super(Critic, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.feature_extractor = feature_extractor
        self.feature_dims = feature_dims

        self.q_nets = []
        for i in range(2):
            q_net = nn.Sequential(
                    nn.Linear(self.feature_dims + self.action_space, 256), nn.ReLU(),
                    nn.Linear(256, 256), nn.ReLU(),
                    nn.Linear(256, 64), nn.ReLU(),
                    nn.Linear(64, 1)
                    )
            self.add_module(f"q_net{i}", q_net)
            self.q_nets.append(q_net)

        self.apply(weight_init)


    def forward(self, obs, actions):
        with torch.no_grad():
            features = self.feature_extractor(obs)

        q_input = torch.cat([features, actions], dim = 1)
        # embed()
        return tuple(q_net(q_input) for q_net in self.q_nets)

    def q1_forward(self, obs, actions):
        with torch.no_grad():
            features = self.feature_extractor(obs)

        return self.q_nets[0](torch.cat([features, actions], dim = 1))

    def get_parameters(self):
        parameters = [params for name, params in self.named_parameters() if "feature_extractor" not in name]
        return parameters

