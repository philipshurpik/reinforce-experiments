# Inspired by https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from collections import namedtuple
from pytorch.reinforce import Policy, ReinforceBrain
import numpy as np

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class PolicyA2C(Policy):
    def __init__(self, n_states, n_actions, n_hidden):
        super(PolicyA2C, self).__init__(n_states, n_actions, n_hidden)
        self.V2 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        z1 = self.W1(x)
        a1 = F.relu(z1)
        z2 = self.W2(a1)
        aprob = F.softmax(z2, dim=1)
        state_value = self.V2(a1)
        return aprob, state_value


class A2CBrain(ReinforceBrain):
    def __init__(self, seed, n_states, n_actions, n_hidden):
        super(A2CBrain, self).__init__(seed, n_states, n_actions, n_hidden)

    def make_model(self, seed, n_states, n_actions, n_hidden):
        return PolicyA2C(n_states, n_actions, n_hidden)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        aprob, state_value = self.model(Variable(state))
        m = Categorical(aprob)
        action = m.sample()
        self.model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.data[0].numpy()

    def compute_loss(self, rewards):
        policy_losses = []
        value_losses = []
        for (log_prob, value), r in zip(self.model.saved_actions, rewards):
            reward = r - value.data[0, 0]
            policy_losses.append(-log_prob * reward)
            value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r])).unsqueeze(0)).unsqueeze(0))
        return torch.cat(policy_losses,0).sum() + torch.cat(value_losses,0).sum()








