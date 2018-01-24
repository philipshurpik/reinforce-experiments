import argparse
import gym
from pytorch.reinforce import ReinforceBrain
from pytorch.a2c import A2CBrain
from np.reinforce import ReinforceBrain as NPReinforceBrain
from tf.reinforce import ReinforceBrain as TFReinforceBrain

parser = argparse.ArgumentParser(description='Policy gradients algorithms examples')
parser.add_argument('--type',
                    default='pytorch',
                    choices=['numpy', 'pytorch', 'tensorflow'],
                    help='choice type - numpy or pytorch or tensorflow')
parser.add_argument('--model',
                    default='reinforce',
                    choices=['reinforce', 'a2c'],
                    help='choice model type - a2c or policy gradient')
parser.add_argument('--env',
                    default='CartPole',
                    choices=['CartPole', 'MountainCar', 'LunarLander'],
                    help='choice env')
args = parser.parse_args()

SEED = 42
PRINT_ITERATIONS = 1
MAX_EPISODES = 100000
ENVS = {'CartPole': 'CartPole-v0', 'MountainCar': 'MountainCar-v0', 'LunarLander': 'LunarLander-v2'}
MAX_STEPS = {'MountainCar-v0': 2500, 'CartPole-v0': 2500, 'LunarLander-v2': 500}
IS_SOLVED = {
    'MountainCar-v0': lambda avg_reward: avg_reward < 100,
    'CartPole-v0': lambda avg_reward: avg_reward > 2400,
    'LunarLander-v2': lambda avg_reward: avg_reward > 50
}

ENV_NAME = ENVS[args.env]
env = gym.make(ENV_NAME)
print("Default max episode steps", env._max_episode_steps)
env._max_episode_steps = MAX_STEPS[ENV_NAME]
env.seed(SEED)
render = True
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n
n_hidden = 64

TORCH_MODELS = {'a2c': A2CBrain, 'reinforce': ReinforceBrain}
NUMPY_MODELS = {'reinforce': NPReinforceBrain}
TF_MODELS = {'reinforce': TFReinforceBrain}
MODELS = NUMPY_MODELS if args.type == 'numpy' else TF_MODELS if args.type == 'tensorflow' else TORCH_MODELS
brain = MODELS[args.model](seed=SEED, n_states=n_states, n_actions=n_actions, n_hidden=n_hidden)


def main():
    avg_reward = None
    for i_episode in range(MAX_EPISODES):
        state = env.reset()
        for t in range(MAX_STEPS[ENV_NAME]):
            action = brain.select_action(state)
            state, reward, done, _ = env.step(action)
            if render:
                env.render()
            brain.add_step_reward(reward)
            if done:
                break

        last_reward = brain.get_rewards_sum()
        avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1
        brain.finish_episode()
        if i_episode % PRINT_ITERATIONS == 0:
            print('Episode {}\tLength: {:5d}\tEpisode Reward: {:.2f}\tAverage Reward: {:.2f}'.format(
                i_episode, (t + 1), last_reward, avg_reward))
        if IS_SOLVED[ENV_NAME](avg_reward):
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(avg_reward, (t + 1)))
            break


if __name__ == '__main__':
    main()

