# Inspired by https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5

import numpy as np

gamma = 0.99
n_hidden = 128
learning_rate = 1e-3
decay_rate = 0.99


class Policy:
    def __init__(self, n_states, n_actions):
        n_actions_1 = 1
        self.model = {
            'W1': np.random.randn(n_hidden, n_states) / np.sqrt(n_states),
            'W2': np.random.randn(n_hidden, n_actions_1) / np.sqrt(n_actions_1)
        }
        # update buffers that add up gradients over a batch
        self.grad_buffer = {k: np.zeros_like(v) for k, v in self.model.items()}
        # rmsprop memory
        self.rmsprop_cache = {k: np.zeros_like(v) for k, v in self.model.items()}
        self.rewards = []

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]

    def forward(self, x):
        h = np.dot(self.model['W1'], x)
        h[h < 0] = 0    # ReLU nonlinearity
        logp = np.dot(self.model['W2'].T, h.reshape(-1, 1))
        p = Policy.sigmoid(logp)
        return p, h     # return probability of taking action 2, and hidden state

    def backward(self, x_cache, h_cache, epdlogp):
        """ backward pass. (h_cache is array of intermediate hidden states) """
        dW2 = np.dot(h_cache.T, epdlogp)
        dh = np.outer(epdlogp, self.model['W2'])
        dh[h_cache <= 0] = 0  # backpro prelu
        dW1 = np.dot(dh.T, x_cache)
        return {'W1': dW1, 'W2': dW2}


class ReinforceBrain:
    def __init__(self, seed, n_states, n_actions):
        np.random.seed(seed)
        self.model = self.make_model(n_states, n_actions)
        self.x_cache, self.h_cache, self.dlogps = [], [], []

    def make_model(self, n_states, n_actions):
        return Policy(n_states, n_actions)

    def select_action(self, state):
        aprob, h = self.model.forward(state)
        probs = [1 - aprob[0][0], aprob[0][0]]
        action = np.random.choice(2, p=probs)

        self.x_cache.append(state)  # state
        self.h_cache.append(h)  # hidden state
        self.dlogps.append(action - aprob)  # grad that encourages the action that was taken to be taken
        return action

    def add_step_reward(self, reward):
        self.model.rewards.append(reward)

    def get_rewards_sum(self):
        return np.sum(self.model.rewards)

    def finish_episode(self):
        x_cache = np.vstack(self.x_cache)
        h_cache = np.vstack(self.h_cache)
        discounted_rewards = np.array(self.discount_rewards(self.model.rewards)).reshape(-1, 1)
        epdlogp = np.vstack(self.dlogps) * discounted_rewards

        grad = self.model.backward(x_cache, h_cache, epdlogp)
        for k in self.model.model:
            self.model.grad_buffer[k] += grad[k]

        for k, v in self.model.model.items():
            g = self.model.grad_buffer[k]  # gradient
            self.model.rmsprop_cache[k] = decay_rate * self.model.rmsprop_cache[k] + (1 - decay_rate) * g ** 2
            self.model.model[k] += learning_rate * g / (np.sqrt(self.model.rmsprop_cache[k]) + 1e-5)
            self.model.grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

        self.x_cache, self.h_cache, self.dlogps = [], [], []
        del self.model.rewards[:]

    def discount_rewards(self, model_rewards):
        running_add = 0
        discounted_rewards = []
        for r in model_rewards[::-1]:
            running_add = r + gamma * running_add
            discounted_rewards.insert(0, running_add)
        eps = np.finfo(np.float32).eps
        return (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + eps)
