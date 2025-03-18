import numpy as np
import pickle
import random
import gym

'''
state = (
    taxi_row, 0
    taxi_col, 1
    self.stations[0][0], 2
    self.stations[0][1], 3
    self.stations[1][0], 4
    self.stations[1][1], 5
    self.stations[2][0], 6
    self.stations[2][1], 7
    self.stations[3][0], 8
    self.stations[3][1], 9
    obstacle_north, 10
    obstacle_south, 11
    obstacle_east, 12
    obstacle_west, 13
    passenger_look, 14
    destination_look 15
)
'''

def distance_compression(relative_distance):
    if relative_distance[0] > 0:
        x = 1
    elif relative_distance[0] < 0:
        x = -1
    else:
        x = 0
    if relative_distance[1] > 0:
        y = 1
    elif relative_distance[1] < 0:
        y = -1
    else:
        y = 0
    return (x, y)

'''
returns (
    relative_direction_to_destination -> tuple(int, int), 
    obstacles_of_four_way -> tuple(int, int, int, int), 
    can_pickup -> bool,
    can_drop -> bool,
    
    # below not for table storing
    
    relative_distance_to_destination -> tuple(int, int),
    current_has_passenger -> bool,
    current_heading_destination_index -> int,
    )
'''

def get_obs_state(obs, has_pas=False, current_des_sta=0):
    taxi_row = obs[0]
    taxi_col = obs[1]
    stations = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]
    obstacles = tuple(obs[10:14])
    passenger_look = obs[14]
    destination_look = obs[15]
    current_des = stations[current_des_sta]
    relative_dist = (current_des[0] - taxi_row, current_des[1] - taxi_col)
    if relative_dist[0] == 0 and relative_dist[1] == 0:
        if has_pas:
            if destination_look:
                # has passenger and at destination
                return_value = (
                    distance_compression(relative_dist), obstacles, False, True, 
                    relative_dist, has_pas, current_des_sta)
                return return_value
            else:
                # has passenger but not at destination
                current_des_sta = ((current_des_sta + 1) & 3)
                current_des = stations[current_des_sta]
                relative_dist = (current_des[0] - taxi_row, current_des[1] - taxi_col)
                
                return_value = (
                    distance_compression(relative_dist), obstacles, False, False, 
                    relative_dist, has_pas, current_des_sta)
                return return_value
        else:
            if passenger_look:
                # this block has passenger
                return_value = (
                    distance_compression(relative_dist), obstacles, True, False, 
                    relative_dist, has_pas, current_des_sta)
                return return_value
            else:
                # this block has no passenger
                current_des_sta = ((current_des_sta + 1) & 3)
                current_des = stations[current_des_sta]
                relative_dist = (current_des[0] - taxi_row, current_des[1] - taxi_col)
                
                return_value = (
                    distance_compression(relative_dist), obstacles, False, False, 
                    relative_dist, has_pas, current_des_sta)
                return return_value
    else:
        return_value = (
            distance_compression(relative_dist), obstacles, False, False, 
            relative_dist, has_pas, current_des_sta)
        return return_value

n = 0

def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.

    n = n ^ 1
    return n # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.


