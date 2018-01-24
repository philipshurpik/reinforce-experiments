# Inspired by https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5

import numpy as np

gamma = 0.99
n_hidden = 64
learning_rate = 3e-3
decay_rate = 0.99


class Policy:
    def __init__(self, n_states, n_actions):
        self.model = {
            'W1': np.random.randn(n_states, n_hidden) / np.sqrt(n_states),
            'W2': np.random.randn(n_hidden, n_actions) / np.sqrt(n_actions)
        }
        # update buffers that add up gradients over a batch
        self.grad_buffer = {k: np.zeros_like(v) for k, v in self.model.items()}
        # rmsprop memory
        self.rmsprop_cache = {k: np.zeros_like(v) for k, v in self.model.items()}
        self.rewards, self.x_cache, self.a1_cache, self.dlogprobs = [], [], [], []

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def relu(z): return np.maximum(0, z)

    def forward(self, x):
        z1 = np.dot(x, self.model['W1'])
        a1 = Policy.relu(z1)
        z2 = np.dot(a1, self.model['W2'])
        a_prob = Policy.softmax(z2)
        return a_prob, a1   # return probability of taking actions, and hidden state

    def backward(self, x_cache, a1_cache, dZ2):
        """ backward pass. (a1_cache is array of intermediate hidden states) """
        dW2 = np.dot(a1_cache.T, dZ2)
        dZ1 = np.dot(self.model['W2'], dZ2.T)
        dZ1[a1_cache.T <= 0] = 0
        dW1 = np.dot(x_cache.T, dZ1.T)
        return {'W1': dW1, 'W2': dW2}


class ReinforceBrain:
    def __init__(self, seed, n_states, n_actions):
        np.random.seed(seed)
        self.model = self.make_model(n_states, n_actions)
        self.n_actions = n_actions

    def make_model(self, n_states, n_actions):
        return Policy(n_states, n_actions)

    def select_action(self, state):
        aprob, a1 = self.model.forward(state)
        action = np.random.choice(self.n_actions, p=aprob)
        y = np.zeros_like(aprob)
        y[action] = 1

        self.model.x_cache.append(state)  # state
        self.model.a1_cache.append(a1)  # hidden state
        self.model.dlogprobs.append(y - aprob)  # grad that encourages the action that was taken to be taken
        return action

    def add_step_reward(self, reward):
        self.model.rewards.append(reward)

    def get_rewards_sum(self):
        return np.sum(self.model.rewards)

    def finish_episode(self):
        x_cache = np.vstack(self.model.x_cache)
        a1_cache = np.vstack(self.model.a1_cache)
        discounted_rewards = np.array(self.discount_rewards(self.model.rewards)).reshape(-1, 1)
        dlogprobs_advantage = np.vstack(self.model.dlogprobs) * discounted_rewards

        grad = self.model.backward(x_cache, a1_cache, dlogprobs_advantage)
        for k in self.model.model:
            self.model.grad_buffer[k] += grad[k]

        for k, v in self.model.model.items():
            g = self.model.grad_buffer[k]  # gradient
            self.model.rmsprop_cache[k] = decay_rate * self.model.rmsprop_cache[k] + (1 - decay_rate) * g ** 2
            self.model.model[k] += learning_rate * g / (np.sqrt(self.model.rmsprop_cache[k]) + 1e-5)
            self.model.grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

        self.model.rewards, self.model.x_cache, self.model.a1_cache, self.model.dlogprobs = [], [], [], []

    def discount_rewards(self, model_rewards):
        running_add = 0
        discounted_rewards = []
        for r in model_rewards[::-1]:
            running_add = r + gamma * running_add
            discounted_rewards.insert(0, running_add)
        eps = np.finfo(np.float32).eps
        return (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + eps)
