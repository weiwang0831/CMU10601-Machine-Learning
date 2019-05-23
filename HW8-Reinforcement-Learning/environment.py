import numpy as np
import sys

maze_input = sys.argv[1]
output_file = sys.argv[2]
action_seq_file = sys.argv[3]


# maze_input = "medium_maze.txt"
# output_file = "output.feedback"
# action_seq_file = "medium_maze_action_seq.txt"


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

    def export(self, exportfile, resultlist):
        file = open(exportfile, "w")
        for line in resultlist:
            l = len(line)
            string = ""
            for i in range(0, l):
                if i == 0:
                    string = string + str(line[i])
                elif i < l - 1:
                    string = string + " " + str(line[i])
                elif i == l - 1:
                    string = string + " " + str(line[i]) + "\n"
            file.write(string)
        file.close()

    def reset(self):
        self.now = self.start
        return self.now


envr = Environment(maze_input)
actions = envr.read_action(action_seq_file)
result = []
for ac in actions:
    result.append(envr.step(ac))
envr.export(output_file, result)
