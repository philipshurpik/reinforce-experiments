import numpy as np
import tensorflow as tf
from shared_brain import SharedBrain

learning_rate = 3e-3


class Policy:
    def __init__(self, seed, n_states, n_actions, n_hidden):
        initializer = tf.contrib.layers.xavier_initializer(seed=seed)
        self.model = {
            'W1': tf.Variable(initializer([n_states, n_hidden]), name="W1"),
            'W2': tf.Variable(initializer([n_hidden, n_actions]), name="W2")
        }
        self.rewards, self.x_cache, self.y_cache = [], [], []

    def forward(self, x):
        z1 = tf.matmul(x, self.model['W1'])
        a1 = tf.nn.relu(z1)
        z2 = tf.matmul(a1, self.model['W2'])
        a_prob = tf.nn.softmax(z2)
        return a_prob


class ReinforceBrain(SharedBrain):
    def __init__(self, seed, n_states, n_actions, n_hidden):
        super(ReinforceBrain, self).__init__(seed, n_states, n_actions, n_hidden)
        tf.set_random_seed(seed)
        self.tf_x = tf.placeholder(dtype=tf.float32, shape=[None, n_states], name="x_cache")
        self.tf_y = tf.placeholder(dtype=tf.float32, shape=[None, n_actions], name="y_cache")
        self.tf_disc_rewards = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="disc_rewards")

        self.policy_forward = self.model.forward(self.tf_x)

        loss = tf.nn.l2_loss(self.tf_y - self.policy_forward)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        tf_grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables(), grad_loss=self.tf_disc_rewards)
        self.train_op = optimizer.apply_gradients(tf_grads)

        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def make_model(self, seed, n_states, n_actions, n_hidden):
        return Policy(seed, n_states, n_actions, n_hidden)

    def run_forward(self, state):
        return self.session.run(self.policy_forward, {self.tf_x: np.reshape(state, (1, -1))})

    def select_action(self, state):
        aprob = self.run_forward(state)[0, :]
        action = np.random.choice(self.n_actions, p=aprob)
        y = np.zeros_like(aprob)
        y[action] = 1

        self.model.x_cache.append(state)
        self.model.y_cache.append(y)
        return action

    def finish_episode(self):
        x_cache = np.vstack(self.model.x_cache)
        y_cache = np.vstack(self.model.y_cache)
        discounted_rewards = np.array(self.discount_rewards(self.model.rewards)).reshape(-1, 1)

        feed = {self.tf_x: x_cache, self.tf_y: y_cache, self.tf_disc_rewards: discounted_rewards}
        _ = self.session.run(self.train_op, feed)

        self.model.x_cache, self.model.y_cache, self.model.rewards = [], [], []
