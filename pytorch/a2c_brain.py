# Inspired by https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from collections import namedtuple
from pytorch.policy_brain import Policy, PolicyBrain

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
gamma = 0.99


class PolicyA2C(Policy):
    def __init__(self, n_states, n_actions):
        super(PolicyA2C, self).__init__(n_states, n_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=1), state_values


class A2CBrain(PolicyBrain):
    def __init__(self, seed, n_states, n_actions):
        super(A2CBrain, self).__init__(seed, n_states, n_actions)

    def make_model(self, n_states, n_actions):
        return PolicyA2C(n_states, n_actions)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, state_value = self.model(Variable(state))
        m = Categorical(probs)
        action = m.sample()
        self.model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.data[0]

    def compute_loss(self, rewards):
        policy_losses = []
        value_losses = []
        for (log_prob, value), r in zip(self.model.saved_actions, rewards):
            reward = r - value.data[0, 0]
            policy_losses.append(-log_prob * reward)
            value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r]))))
        return torch.cat(policy_losses).sum() + torch.cat(value_losses).sum()








