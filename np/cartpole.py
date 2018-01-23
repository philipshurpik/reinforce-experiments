""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import gym

# hyperparameters
H = 128  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-3
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
render = True

# model initialization
D = 4
A = 1
model = {}
model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
model['W2'] = np.random.randn(H, A) / np.sqrt(A)

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'].T, h.reshape(-1, 1))
    p = sigmoid(logp)
    return p, h  # return probability of taking action 2, and hidden state


def policy_backward(eph, epdlogp, epx):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp)
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}


env = gym.make('CartPole-v0')
xs, hs, dlogps, drs = [], [], [], []
running_reward = None

MAX_EPISODES = 10000
MAX_STEPS = 500

for i_episode in range(MAX_EPISODES):
    state = env.reset()
    reward_sum = 0
    for t in range(MAX_STEPS):
        # forward the policy network and sample an action from the returned probability
        aprob, h = policy_forward(state)
        #action = np.random.choice(A, p=aprob.flatten()/np.sum(aprob))
        #taken_prob = aprob[action]
        action = 1 if np.random.uniform() < aprob else 0

        # record various intermediates (needed later for backprop)
        xs.append(state)  # state
        hs.append(h)  # hidden state
        dlogps.append(action - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

        state, reward, done, _ = env.step(action)
        if render:
            env.render()
        reward_sum += reward
        drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

        if done:  # an episode finished
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)

            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
            grad = policy_backward(eph, epdlogp, epx)
            for k in model:
                grad_buffer[k] += grad[k]  # accumulate grad over batch

            # perform rmsprop parameter update
            for k, v in model.items():
                g = grad_buffer[k]  # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

            xs, hs, dlogps, drs = [], [], [], []  # reset array memory
            # boring book-keeping
            running_reward = reward_sum if running_reward is None else running_reward * 0.9 + reward_sum * 0.1
            print('Episode {}\tLength: {:5d}\tEpisode Reward: {:.2f}\tAverage Reward: {:.2f}'.format(
                i_episode, 1, reward_sum, running_reward))
            break
