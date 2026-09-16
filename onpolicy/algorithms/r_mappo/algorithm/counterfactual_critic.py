import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_shape_from_obs_space


class CounterfactualCritic(nn.Module):
    """Q_i(S, a_-i, k) for PSO leader-choice credit assignment."""

    def __init__(self, args, cent_obs_space, num_agents, action_dim, device=torch.device("cpu")):
        super(CounterfactualCritic, self).__init__()
        self.num_agents = int(num_agents)
        self.action_dim = int(action_dim)
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(cent_obs_space)
        obs_dim = int(np.prod(obs_shape))
        hidden_size = int(getattr(args, "cf_hidden_size", 128))
        input_dim = obs_dim + self.num_agents * self.action_dim + self.num_agents + self.action_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.to(device)

    def forward(self, cent_obs, joint_actions, agent_ids, candidate_actions):
        cent_obs = self._as_tensor(cent_obs).to(**self.tpdv)
        joint_actions = self._as_tensor(joint_actions).long().to(device=cent_obs.device)
        agent_ids = self._as_tensor(agent_ids).long().to(device=cent_obs.device).view(-1)
        candidate_actions = self._as_tensor(candidate_actions).long().to(device=cent_obs.device).view(-1)

        cent_obs = cent_obs.view(cent_obs.shape[0], -1)
        joint_onehot = torch.nn.functional.one_hot(
            joint_actions,
            num_classes=self.action_dim,
        ).float().view(joint_actions.shape[0], -1)
        agent_onehot = torch.nn.functional.one_hot(agent_ids, num_classes=self.num_agents).float()
        candidate_onehot = torch.nn.functional.one_hot(candidate_actions, num_classes=self.action_dim).float()
        x = torch.cat([cent_obs, joint_onehot, agent_onehot, candidate_onehot], dim=-1)
        return self.net(x)

    @staticmethod
    def _as_tensor(value):
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        return value
