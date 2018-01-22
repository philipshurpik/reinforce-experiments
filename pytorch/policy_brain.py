# Inspired by https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from collections import namedtuple

SavedAction = namedtuple('SavedAction', ['log_prob'])
gamma = 0.99


class Policy(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Policy, self).__init__()
        self.hidden1 = nn.Linear(n_states, 128)
        self.action_head = nn.Linear(128, n_actions)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        action_scores = self.action_head(x)
        return F.softmax(action_scores, dim=1)


class PolicyBrain:
    def __init__(self, seed, n_states, n_actions):
        torch.manual_seed(seed)
        self.model = self.make_model(n_states, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2)

    def make_model(self, n_states, n_actions):
        return Policy(n_states, n_actions)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.model(Variable(state))
        m = Categorical(probs)
        action = m.sample()
        self.model.saved_actions.append(SavedAction(m.log_prob(action)))
        return action.data[0]

    def add_step_reward(self, reward):
        self.model.rewards.append(reward)

    def get_rewards_sum(self):
        return np.sum(self.model.rewards)

    def finish_episode(self):
        R = 0
        rewards = []
        for r in self.model.rewards[::-1]:
            R = r + gamma * R
            rewards.insert(0, R)
        rewards = torch.Tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        self.optimizer.zero_grad()
        loss = self.compute_loss(rewards)
        loss.backward()
        self.optimizer.step()
        del self.model.rewards[:]
        del self.model.saved_actions[:]

    def compute_loss(self, rewards):
        policy_losses = []
        for (log_prob,), reward in zip(self.model.saved_actions, rewards):
            policy_losses.append(-log_prob * reward)
        return torch.cat(policy_losses).sum()









