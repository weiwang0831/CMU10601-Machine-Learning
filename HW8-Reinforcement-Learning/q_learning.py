import numpy as np
import sys
import random

# from environment import Environment

# maze_input = sys.argv[1]
# value_file = sys.argv[2]
# q_value_file = sys.argv[3]
# policy_file = sys.argv[4]
# num_episodes = sys.argv[5]
# max_episode_length = sys.argv[6]
# learning_rate = sys.argv[7]
# discount_factor = sys.argv[8]
# epsilon = sys.argv[9]

maze_input = "medium_maze.txt"
value_file = "value_output.txt"
q_value_file = "q_value_output.txt"
policy_file = "policy_output.txt"
num_episodes = "1000"
max_episode_length = "20"
learning_rate = "0.8"
discount_factor = "0.9"
epsilon = "0.05"

import time

start_time = time.time()


class Environment:
    def __init__(self, filename):
        self.maze_array, self.start, self.goal = self.read_file(filename)
        self.now = self.start

    def read_file(self, filename):
        with open(filename, "r") as doc:
            file_list = []
            for line in doc:
                line = line.strip()
                file_list.append(line)
            file_array = np.array(file_list)
        start = (len(file_array) - 1), 0
        goal = 0, (len(file_array[0]) - 1)
        return file_array, start, goal

    def read_action(self, actionfile):
        with open(actionfile, "r") as doc:
            for line in doc:
                action_list = line.split(" ")
        return action_list

    def step(self, a):
        a = int(a)
        if self.now == self.goal:
            res_x, res_y = self.now
            return res_x, res_y, 0, 1
        else:
            x, y = self.now
            x_tmp = x
            y_tmp = y
            if a == 0 and y != 0:
                y_tmp = y - 1
            elif a == 1 and x != 0:
                x_tmp = x - 1
            elif a == 2 and y != (len(self.maze_array[0]) - 1):
                y_tmp = y + 1
            elif a == 3 and x != (len(self.maze_array) - 1):
                x_tmp = x + 1
            next_state = self.maze_array[x_tmp][y_tmp]
            if next_state == "*":
                self.now = self.now
            else:
                self.now = (x_tmp, y_tmp)
            res_x, res_y = self.now
            return res_x, res_y, -1, 0

    def reset(self):
        self.now = self.start
        return self.now


num_episodes = int(num_episodes)
max_episode_length = int(max_episode_length)
learning_rate = float(learning_rate)
discount_factor = float(discount_factor)
epsilon = float(epsilon)

envr = Environment(maze_input)
maze_array = envr.maze_array
length = len(maze_array)
width = len(maze_array[0])
# initiate the Q table with all zero, all x,y
# every Q value will store the list with q-value for 4 actions
actionlist = [0.0, 0.0, 0.0, 0.0]
Q = np.array([[actionlist] * width] * length)
for x in range(0, length):
    for y in range(0, width):
        if maze_array[x][y] == "*":
            Q[x][y] = [None, None, None, None]
        else:
            Q[x][y] = [0.0, 0.0, 0.0, 0.0]
P = np.array([[0.0] * width] * length)
V = np.array([[0.0] * width] * length)


def select_action(epsilon, now):
    r = random.uniform(0, 1)
    if epsilon > 0 and r < epsilon:
        a = random.choice([0, 1, 2, 3])
    else:  # get the max Q value of next state index as action
        x, y = now
        action_list = Q[x][y]
        a = np.argmax(action_list)
    return a


steps = []
for eps in range(0, num_episodes):
    envr.reset()
    now = envr.now
    for l in range(0, max_episode_length):
        print(str(now[0])+" "+str(now[1]))
        action = select_action(epsilon, now)
        next_state = envr.step(action)
        next_x = next_state[0]
        next_y = next_state[1]
        next_state_q_list = Q[next_x][next_y]
        max_q = max(next_state_q_list)
        Q[now[0]][now[1]][action] = (1 - learning_rate) * Q[now[0]][now[1]][action] \
                                    + learning_rate * (next_state[2] + discount_factor * max_q)
        if maze_array[next_x][next_y] == "*":
            break
        if next_state[3] == 1:
            steps.append(l)
            print(l)
            break
        now = next_state[0], next_state[1]
    for x in range(0, length):
        for y in range(0, width):
            if maze_array[x][y] == "*":
                V[x][y] = None
                P[x][y] = None
            else:
                V[x][y] = np.max(Q[x][y])
                P[x][y] = np.argmax(Q[x][y])
print(maze_array)
# print(Q)
# print(P)
print(V)

# export three files
qfile = open(q_value_file, "w")
pfile = open(policy_file, "w")
vfile = open(value_file, "w")
for x in range(0, length):
    for y in range(0, width):
        q_string = ""
        q_list = Q[x][y]
        if np.isnan(V[x][y]) == False:
            v_string = str(x) + " " + str(y) + " " + str(V[x][y]) + "\n"
            vfile.write(v_string)
        if np.isnan(V[x][y]) == False:
            p_string = str(x) + " " + str(y) + " " + str(P[x][y]) + "\n"
            pfile.write(p_string)
        for i in range(0, len(q_list)):
            if np.isnan(q_list[i]) == False:
                q_string = str(x) + " " + str(y) + " " + str(i) + " " + str(q_list[i]) + "\n"
                qfile.write(q_string)
qfile.close()
vfile.close()
pfile.close()

print(sum(steps))
print(len(steps))
# print(sum(steps) / len(steps))
print("--- %s seconds ---" % (time.time() - start_time))
# print(V[0][0])
# print(V[7][0])
# print(V[6][7])
# print(P[0][2])
# print(P[2][0])
# print(P[2][4])


matrix = [
    [5, 9, 2],
    [8, 1, 7],
    [6, 3, 4]
]
print(matrix[1][2])