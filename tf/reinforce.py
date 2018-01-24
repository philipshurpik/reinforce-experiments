import numpy as np
import tensorflow as tf

gamma = 0.99
n_hidden = 64
learning_rate = 1e-2


class Policy:
    def __init__(self, seed, n_states, n_actions):
        initializer = tf.contrib.layers.xavier_initializer(seed=seed)
        self.model = {
            'W1': tf.Variable(initializer([n_states, n_hidden]), name="W1"),
            'W2': tf.Variable(initializer([n_hidden, n_actions]), name="W2")
        }
        self.rewards = []

    def forward(self, x):
        z1 = tf.matmul(x, self.model['W1'])
        a1 = tf.nn.relu(z1)
        z2 = tf.matmul(a1, self.model['W2'])
        a_prob = tf.nn.softmax(z2)
        return a_prob


class ReinforceBrain:
    def __init__(self, seed, n_states, n_actions):
        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.n_actions = n_actions
        self.model = self.make_model(seed, n_states, n_actions)
        self.tf_x = tf.placeholder(dtype=tf.float32, shape=[None, n_states], name="x_cache")
        self.tf_dlogprobs = tf.placeholder(dtype=tf.float32, shape=[None, n_actions], name="dlogprobs")
        self.tf_disc_rewards = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="disc_rewards")

        self.policy_forward = self.model.forward(self.tf_x)

        loss = tf.nn.l2_loss(self.tf_dlogprobs - self.policy_forward)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        tf_grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables(), grad_loss=self.tf_disc_rewards)
        self.train_op = optimizer.apply_gradients(tf_grads)

        self.session = tf.InteractiveSession()
        tf.initialize_all_variables().run()
        self.x_cache, self.dlogprobs, self.y_cache = [], [], []

    def make_model(self, seed, n_states, n_actions):
        return Policy(seed, n_states, n_actions)

    def run_forward(self, state):
        return self.session.run(self.policy_forward, {self.tf_x: np.reshape(state, (1, -1))})

    def select_action(self, state):
        aprob = self.run_forward(state)[0, :]
        action = np.random.choice(self.n_actions, p=aprob)
        y = np.zeros_like(aprob)
        y[action] = 1

        self.x_cache.append(state)
        self.y_cache.append(y)
        self.dlogprobs.append(y - aprob)
        return action

    def add_step_reward(self, reward):
        self.model.rewards.append(reward)

    def get_rewards_sum(self):
        return np.sum(self.model.rewards)

    def finish_episode(self):
        x_cache = np.vstack(self.x_cache)
        y_cache = np.vstack(self.y_cache)
        dlogprobs = np.vstack(self.dlogprobs)
        discounted_rewards = np.array(self.discount_rewards(self.model.rewards)).reshape(-1, 1)

        feed = {self.tf_x: x_cache, self.tf_dlogprobs: y_cache, self.tf_disc_rewards: discounted_rewards}
        _ = self.session.run(self.train_op, feed)

        self.x_cache, self.dlogprobs, self.y_cache = [], [], []
        del self.model.rewards[:]

    def discount_rewards(self, model_rewards):
        running_add = 0
        discounted_rewards = []
        for r in model_rewards[::-1]:
            running_add = r + gamma * running_add
            discounted_rewards.insert(0, running_add)
        eps = np.finfo(np.float32).eps
        return (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + eps)
