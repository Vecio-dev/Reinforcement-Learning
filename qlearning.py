from os import stat
import gym
import numpy as np

#, render_mode="human"
env = gym.make("MountainCar-v0", render_mode="rgb_array")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 26001

RECORD_EVERY = 2000

# Position - Velocity
# High: [0.6  0.07] 
# Low:  [-1.2  -0.07]
print(env.observation_space.high)
print(env.observation_space.low)
# Number of actions available (3): move left, move right, do nothing 
print(env.action_space.n)

# [20, 20] Position and Velocity would be devided into 20 discrete bins
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)

# Size of each bin: [0.09, 0.007]
# Ex. for position: 0.6 - (-1.2) = 1.8 which is the range values
# therefore, the size of each bin in the position dimension would be 1.8 / 20 = 0.09 units
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# NumPy array which represents the Q-Table
# The size of the array would be [20, 20, 3], and the values are initialized to a random number between -2 and 0
# in the end the Q-table will have 20 x 20 x 3 entries, one for each combination of the discretized states in the observation space and the available actions in the environment.
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Takes in a continuous state and returns a corresponding discrete state, which is a tuple of integers representing the indices of the discrete bins in each dimension of the observation space
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int32))

for episode in range(EPISODES):
    #print(episode)

    if episode % RECORD_EVERY == 0:
        print(f"Reached episode {episode}")
        env = gym.wrappers.RecordVideo(env, 'videos', episode_trigger = lambda x: x == RECORD_EVERY, name_prefix = f"{episode}")

    discrete_state = get_discrete_state(env.reset()[0])
    done = False
    truncated = False
    while not done and not truncated:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        action = np.argmax(q_table[discrete_state])

        new_state, reward, done, truncated, info = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        if not done and not truncated:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            print(f"Completed on episode {episode}")
            q_table[discrete_state + (action, )] = 0
        
        discrete_state = new_discrete_state
    
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

env.close()