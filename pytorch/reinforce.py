# Inspired by https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from collections import namedtuple
from shared_brain import SharedBrain

SavedAction = namedtuple('SavedAction', ['log_prob'])
learning_rate = 3e-3


class Policy(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(Policy, self).__init__()
        self.hidden1 = nn.Linear(n_states, n_hidden)
        self.action_head = nn.Linear(n_hidden, n_actions)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        action_scores = self.action_head(x)
        return F.softmax(action_scores, dim=1)


class ReinforceBrain(SharedBrain):
    def __init__(self, seed, n_states, n_actions, n_hidden):
        super(ReinforceBrain, self).__init__(seed, n_states, n_actions, n_hidden)
        torch.manual_seed(seed)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def make_model(self, seed, n_states, n_actions, n_hidden):
        return Policy(n_states, n_actions, n_hidden)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.model(Variable(state))
        m = Categorical(probs)
        action = m.sample()
        self.model.saved_actions.append(SavedAction(m.log_prob(action)))
        return action.data[0]

    def finish_episode(self):
        rewards = self.discount_rewards(self.model.rewards)
        rewards = torch.Tensor(rewards)
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
