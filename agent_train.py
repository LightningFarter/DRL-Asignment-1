import numpy as np
import pickle
import random
import time
from simple_custom_taxi_env import SimpleTaxiEnv


def save_q_table(q_table, file_path='q_table.pkl'):
    with open(file_path, 'wb') as f:
        pickle.dump(f)

def load_q_table(file_path='q_table.pkl'):
    with open(file_path, 'rb') as f:
        loaded_q_table = pickle.load(f)
    return loaded_q_table

alpha = 0.1
gamma = 0.9999
epsilon = 0.1
num_episodes = 5000

q_table = {}

def train_agent():
    env = SimpleTaxiEnv(grid_size=5, fuel_limit=5000)