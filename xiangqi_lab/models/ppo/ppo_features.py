"""Extract board features and pool the complete bounded repetition table."""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class HistoryFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=128)
        self.board_net = nn.Sequential(nn.Flatten(), nn.Linear(1260, 128), nn.ReLU())
        self.history_net = nn.Sequential(nn.Linear(92, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.output_net = nn.Sequential(nn.Linear(128 + 64 + 6, 128), nn.ReLU())

    def forward(self, observations):
        history = observations['history']
        mask = (history[:, :, -1:] > 0).float()
        rows = self.history_net(history) * mask
        mean = rows.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        maximum = rows.max(dim=1).values
        return self.output_net(torch.cat([
            self.board_net(observations['board']), mean, maximum, observations['stats'],
        ], dim=1))
