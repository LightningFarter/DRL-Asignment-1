import numpy as np
import pickle
import random
from simple_custom_taxi_env import SimpleTaxiEnv  # Assuming the environment code is in `simple_taxi_env.py`


ALPHA = 0.1
GAMMA = 0.99999
EPSILON = 0.1
EPISODES = 1000000
ACTION_SPACE = [0, 1, 2, 3, 4, 5]

policy_table = {}

def get_action(state):
    if state not in policy_table:
        policy_table[state] = [1 / len(ACTION_SPACE)] * len(ACTION_SPACE)
    
    if random.random() < EPSILON:
        return random.choice(ACTION_SPACE)
    else:
        return np.argmax(policy_table[state])

def update_policy_table(state, action, reward, next_state, done):
    if next_state not in policy_table:
        policy_table[next_state] = [1 / len(ACTION_SPACE)] * len(ACTION_SPACE)
    
    max_next_value = max(policy_table[next_state]) if not done else 0
    target = reward + GAMMA * max_next_value
    
    current_probs = policy_table[state]
    current_value = current_probs[action]
    updated_value = current_value + ALPHA * (target - current_value)
    
    current_probs[action] = updated_value
    total = sum(current_probs)
    policy_table[state] = [prob / total for prob in current_probs]

def train_agent():
    env = SimpleTaxiEnv(grid_size=10, fuel_limit=50)

    for episode in range(EPISODES):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = get_action(state)

            next_state, reward, done, _ = env.step(action)

            update_policy_table(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward}")

    with open("policy_table.pkl", "wb") as f:
        pickle.dump(policy_table, f)
    print("Training complete. Policy table saved to 'policy_table.pkl'.")

if __name__ == "__main__":
    train_agent()
