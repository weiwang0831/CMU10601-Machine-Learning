import numpy as np
import sys
import time

maze_input = sys.argv[1]
value_file = sys.argv[2]
q_value_file = sys.argv[3]
policy_file = sys.argv[4]
num_epoch = sys.argv[5]
discount_factor = sys.argv[6]

# maze_input = "tiny_maze.txt"
# value_file = "value_output.txt"
# q_value_file = "q_value_output.txt"
# policy_file = "policy_output.txt"
# num_epoch = "5"
# discount_factor = "0.9"

start_time = time.time()


def read_file(filename):
    with open(filename, "r") as doc:
        file_list = []
        for line in doc:
            file_list.append(line)
        file_array = np.array(file_list)
    return file_array


def value_iteration(file_array, action):
    v_table = np.array([[0.0] * (len(file_array[0]) - 1)] * len(file_array))
    count = 0
    for i in range(1, int(num_epoch) + 1):
        v_table = cal_V(v_table, file_array, action)[0]
        # Vdiff_list = cal_V(v_table, file_array, action)[1]
        count = count + 1
        # max_list = []
        print(i)
        print(v_table)
        # for l in Vdiff_list:
        #     max_list.append(max(abs(n) for n in l))
        # if max(max_list) < 0.001:
        #     break
    print(count)
    return v_table


def cal_V(vtable, file_array, action):
    Vdiff = np.array([[0.0] * (len(file_array[0]) - 1)] * len(file_array))
    for x in range(0, len(vtable)):
        for y in range(0, len(vtable[x])):
            cur_state = file_array[x][y]
            cur_V = vtable[x][y]
            Q_list = []
            if cur_state == "G":
                vtable[x][y] = 0
            elif cur_state != "*":
                for a in action:
                    x_tmp = x
                    y_tmp = y
                    if (a == 0 and y != 0):
                        y_tmp = y - 1
                    elif (a == 1 and x != 0):
                        x_tmp = x - 1
                    elif (a == 2 and y != (len(vtable[x]) - 1)):
                        y_tmp = y + 1
                    elif (a == 3 and x != (len(vtable) - 1)):
                        x_tmp = x + 1
                    next_state = file_array[x_tmp][y_tmp]
                    if next_state == "*":
                        next_V = cur_V
                    elif next_state == "G":
                        next_V = 0
                    else:
                        next_V = vtable[x_tmp][y_tmp]
                    Q = float(-1) + float(discount_factor) * float(next_V)
                    Q_list.append(Q)
                vtable[x][y] = max(Q_list)
            Vdiff[x][y] = np.abs(vtable[x][y] - cur_V)
    return [vtable, Vdiff]


def cal_Q(vtable, file_array, action):
    actionlist = [0.0, 0.0, 0.0, 0.0]
    total_Q = np.array([[actionlist] * len(file_array[0])] * len(file_array))
    total_policy = np.array([[0.0] * len(file_array[0])] * len(file_array))
    for x in range(0, len(vtable)):
        for y in range(0, len(vtable[x])):
            cur_state = file_array[x][y]
            cur_V = vtable[x][y]
            if cur_state == "G":
                vtable[x][y] = 0
                Q_list = [0.0, 0.0, 0.0, 0.0]
                policy = np.argmax(np.array(Q_list))
                total_Q[x][y] = Q_list
                total_policy[x][y] = float(policy)
            elif cur_state != "*":
                Q_list = [0.0, 0.0, 0.0, 0.0]
                for a in action:
                    x_tmp = x
                    y_tmp = y
                    if (a == 0 and y != 0):
                        y_tmp = y - 1
                    elif (a == 1 and x != 0):
                        x_tmp = x - 1
                    elif (a == 2 and y != (len(vtable[x]) - 1)):
                        y_tmp = y + 1
                    elif (a == 3 and x != (len(vtable) - 1)):
                        x_tmp = x + 1
                    next_state = file_array[x_tmp][y_tmp]
                    if next_state == "*":
                        next_V = cur_V
                    elif next_state == "G":
                        next_V = 0.0
                    else:
                        next_V = vtable[x_tmp][y_tmp]
                    Q = float(-1) + float(discount_factor) * float(next_V)
                    Q_list[a] = Q
                policy = np.argmax(np.array(Q_list))
                total_Q[x][y] = Q_list
                total_policy[x][y] = float(policy)
    return [total_Q, total_policy]


def export(V, Q, P, maze):
    file = open(value_file, "w")
    file1 = open(policy_file, "w")
    file2 = open(q_value_file, "w")
    for x in range(0, len(V)):
        for y in range(0, len(V[0])):
            cur_state = maze[x][y]
            if cur_state != "*":
                file.write(str(x) + " " + str(y) + " " + str(V[x][y]) + "\n")
                file1.write(str(x) + " " + str(y) + " " + str(P[x][y]) + "\n")
                for a in action:
                    string = str(x) + " " + str(y) + " " + str(a) + " " + str(Q[x][y][a]) + "\n"
                    file2.write(string)
    file2.close()
    file.close()
    file1.close()


maze_array = read_file(maze_input)
action = [0, 1, 2, 3]
v_table = value_iteration(maze_array, action)
q_table = cal_Q(v_table, maze_array, action)[0]
policy_table = cal_Q(v_table, maze_array, action)[1]
# print(v_table)
# print(v_table[0][0])
# print(v_table[7][0])
# print(v_table[6][7])
# print(q_table)
# print(policy_table)
# print(policy_table[2][0])
# print(policy_table[0][2])
# print(policy_table[2][4])
export(v_table, q_table, policy_table, maze_array)

print("--- %s seconds ---" % (time.time() - start_time))
