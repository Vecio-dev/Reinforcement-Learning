from os import stat
import gym
import numpy as np

#, render_mode="human"
env = gym.make("MountainCar-v0", render_mode="rgb_array")

# A hyperparameter (0-1) that controls the extent to which the new Q-value should be based on the observed reward versus the current estimate of the Q-value. A high learning rate means that the new Q-value is heavily influenced by the observed reward, while a low learning rate means that the new Q-value is closer to the current estimate of the Q-value.
LEARNING_RATE = 0.1
# A hyperparameter (0-1) that controls the extent to which future rewards should be discounted. A high discount factor means that the agent cares more about immediate rewards, while a low discount factor means that the agent cares more about long-term rewards.
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

    # Record a video of the episode every RECORD_EVERY episodes
    if episode % RECORD_EVERY == 0:
        print(f"Reached episode {episode}")
        env = gym.wrappers.RecordVideo(env, 'videos', episode_trigger = lambda x: x == RECORD_EVERY, name_prefix = f"{episode}")

    # Reset the environment every new episode and get its discrete state
    discrete_state = get_discrete_state(env.reset()[0])

    done = False
    truncated = False
    while not done and not truncated:
        if np.random.random() > epsilon:
            # Returns the index of the highest Q-value in the q_table[discrete_state], which represents the action with the highest expected reward according to the current policy
            action = np.argmax(q_table[discrete_state])
        else:
            # Returns a random number between 0 and 3 for the agent to explore the environment
            action = np.random.randint(0, env.action_space.n)

        # Take the action and get its new discrete state
        new_state, reward, done, truncated, info = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        # If the agent dind't manage to reach the objective of the environment:
        if not done and not truncated:
            # Get the maximum Q-value for the next state new_discrete_state
            # It represents the maximum expected reward that the agent can obtain from the next state onwards, following the current policy
            max_future_q = np.max(q_table[new_discrete_state])
            # Get the current Q-value for the current state and action taken
            current_q = q_table[discrete_state + (action, )]

            # The new Q-value is calculated as a weighted average of the current Q-value and the maximum expected future reward that can be obtained by following the current policy from the next state onwards. The new Q-value represents the expected total reward that can be obtained by following the current policy from the current state-action pair. The learning rate LEARNING_RATE controls the balance between exploration and exploitation, as it determines how much the new Q-value should be influenced by the observed reward versus the current estimate of the Q-value. The discount factor DISCOUNT allows the agent to prioritize immediate rewards over long-term rewards, by reducing the value of future rewards.
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            # Set the new Q-value of the current discrete state and action
            q_table[discrete_state + (action, )] = new_q
        # If the agent reached the objective of the environment:
        elif new_state[0] >= env.goal_position:
            print(f"Completed on episode {episode}")
            # Update the current discrete state and action to be the best that could be taken
            q_table[discrete_state + (action, )] = 0
        
        # Update the current discrete state before changing episode
        discrete_state = new_discrete_state
    
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

# Close the environment when all the episodes finished
env.close()