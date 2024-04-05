import numpy as np

goal_pos = np.array([0,0,2])
current_pos = np.array([0.49,0,2])
bonus_point = np.array([np.array([i,0,2]) for i in range(int(goal_pos[0]+1))])

goal_pos_x = int(goal_pos[0])
bonus = 0.1*(1-np.exp(-np.min([np.abs(round(current_pos[0])), goal_pos_x])))
if np.linalg.norm(current_pos - bonus_point[np.min([np.abs(round(current_pos[0])), goal_pos_x])]) < 1.0: 
    # total_reward += bonus
    if bonus >= 0:
        print("Bonus earned  |  Bonus Point: {bp}  |  Postion: {pos}  |  Bonus: {bonus}".format(
                bp=bonus_point[np.min([np.abs(round(current_pos[0])), goal_pos_x])],
                pos=np.round(current_pos, 2),
                bonus=np.round(bonus, 2)))