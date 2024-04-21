import numpy as np
from collections import deque

goal_pos = np.array([5,0,2])
current_pos = np.array([4.3,0,2])
bonus_point = np.array([np.array([i,0,2]) for i in range(int(goal_pos[0]+1))])
previous_epi_len = deque([6000,6000,6000,6000])
goal_pos_x = int(goal_pos[0])

print(bonus_point)
print(np.average(previous_epi_len))
print(np.exp(-np.average(previous_epi_len)/10000)) # radius

print(np.linalg.norm(current_pos - bonus_point[np.min([np.abs(round(current_pos[0])), goal_pos_x])]))


bonus = 0.1*(1-np.exp(-np.min([np.abs(round(current_pos[0])), goal_pos_x])))
if np.linalg.norm(current_pos - bonus_point[np.min([np.abs(round(current_pos[0])), goal_pos_x])]) < np.exp(-np.average(previous_epi_len)/10000):
    # total_reward += bonus
    if bonus > 0:
        print("Bonus earned  |  Bonus Point: {bp}  |  Postion: {pos}  |  Bonus: {bonus}".format(
                bp=bonus_point[np.min([np.abs(round(current_pos[0])), goal_pos_x])],
                pos=np.round(current_pos, 2),
                bonus=np.round(bonus, 3)))