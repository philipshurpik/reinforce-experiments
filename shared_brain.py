import numpy as np

gamma = 0.99


class SharedBrain:
    def __init__(self, seed, n_states, n_actions, n_hidden):
        np.random.seed(seed)
        self.n_actions = n_actions
        self.model = self.make_model(seed, n_states, n_actions, n_hidden)

    def make_model(self, seed, n_states, n_actions, n_hidden):
        raise NotImplementedError

    def select_action(self, state):
        raise NotImplementedError

    def add_step_reward(self, reward):
        self.model.rewards.append(reward)

    def get_rewards_sum(self):
        return np.sum(self.model.rewards)

    def finish_episode(self):
        raise NotImplementedError

    def discount_rewards(self, model_rewards):
        running_add = 0
        discounted_rewards = []
        for r in model_rewards[::-1]:
            running_add = r + gamma * running_add
            discounted_rewards.insert(0, running_add)
        eps = np.finfo(np.float32).eps
        return (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + eps)
