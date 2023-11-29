import random
import gym
import numpy as np
import matplotlib.pyplot as plt



def init_q_table(x: int, y: int) -> np.ndarray:
    return np.zeros((x, y))

qTable = init_q_table(5, 4)

# print("Q-Table:\n" + str(qTable))
assert np.mean(qTable) == 0


LEARNING_RATE = 0.05
DISCOUNT_RATE = 0.99

def q_function(q_table: np.ndarray, state: int, action: int, reward: int, new_state: int) -> np.ndarray:
    current_q = q_table[state, action]
    max_future_q = np.max(q_table[new_state, :])

    updated_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_RATE * max_future_q)

    q_table[state, action] = updated_q

    return q_table

q_table = init_q_table(5, 4)

q_table = q_function(q_table, state=0, action=1, reward=-1, new_state=3)

# print("Q-Table after action:\n" + str(q_table))

assert np.isclose(q_table[0, 1], -LEARNING_RATE), f"The Q function is incorrect: the value of q_table[0, 1] should be approximately -{LEARNING_RATE}"
env = gym.make('FrozenLake-v1', is_slippery=False)


print("Action space:", env.action_space)
print("Observation space:", env.observation_space)
total_actions = env.action_space.n

print("Total number of actions:", total_actions)

def random_action(env):
    total_actions = env.action_space.n
    action = np.random.randint(total_actions)
    return action

def game_loop(env: gym.Env, q_table: np.ndarray, state: int, action: int) -> tuple:
    # Perform the selected action
    new_state, reward, done, _ = env.step(action)

    # Update the Q-table using the q_function
    q_table[state, action] = q_function(q_table, state, action, reward, new_state)

    return q_table, new_state, reward, done

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")
q_table = init_q_table(env.observation_space.n, env.action_space.n)

state, info = env.reset()
while True:
    env.render()
    action = random_action(env)
    q_table, new_state, reward, done = game_loop(env, q_table, state, action)
    if done:
        break
env.close()