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

with open('policy_table.pkl', 'rb') as f:
    policy_table = pickle.load(f)

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

current_has_passenger = False
current_destination_index = 0

action_space = [0, 1, 2, 3, 4, 5]
last_action = -1

def get_obs_state(obs, has_pas=False, current_des_sta=0):
    taxi_row = obs[0]
    taxi_col = obs[1]
    stations = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[8], obs[9]), (obs[6], obs[7])]
    obstacles = (obs[11], obs[10], obs[12], obs[13]) # tuple(obs[10:14])
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

def get_action(obs):
    global current_has_passenger
    global current_destination_index
    global last_action
    
    if last_action == 4 and obs[14] == 1:
        current_has_passenger = True
    elif last_action == 5:
        current_has_passenger = False
    
    rdd, ofw, cp, cd, _, chp, chd = get_obs_state(obs, current_has_passenger, current_destination_index)
    
    state = (rdd, ofw, cp, cd)
    print(state)
    if state in policy_table:
        action = np.random.choice(action_space, p=policy_table[state])
        print(f"action = {action}")
    else:
        print("not in policy")
        action = np.random.choice([0, 1, 2, 3])
    
    current_has_passenger = chp
    current_destination_index = chd
    
    last_action = action
    
    return action

