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
n_hidden = 64
gamma = 0.99


class Policy(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Policy, self).__init__()
        self.hidden1 = nn.Linear(n_states, n_hidden)
        self.action_head = nn.Linear(n_hidden, n_actions)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        action_scores = self.action_head(x)
        return F.softmax(action_scores, dim=1)


class ReinforceBrain:
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
        rewards = self.discount_rewards(self.model.rewards)
        rewards = torch.Tensor(rewards)
        self.optimizer.zero_grad()
        loss = self.compute_loss(rewards)
        loss.backward()
        self.optimizer.step()
        del self.model.rewards[:]
        del self.model.saved_actions[:]

    def discount_rewards(self, model_rewards):
        running_add = 0
        discounted_rewards = []
        for r in model_rewards[::-1]:
            running_add = r + gamma * running_add
            discounted_rewards.insert(0, running_add)
        eps = np.finfo(np.float32).eps
        return (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + eps)

    def compute_loss(self, rewards):
        policy_losses = []
        for (log_prob,), reward in zip(self.model.saved_actions, rewards):
            policy_losses.append(-log_prob * reward)
        return torch.cat(policy_losses).sum()









