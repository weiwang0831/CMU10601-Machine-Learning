import sys
import numpy as np
import math
import matplotlib.pyplot as plt

# formatted_train_input = sys.argv[1]
# formatted_validation_input = sys.argv[2]
# formatted_test_input = sys.argv[3]
# dict_input = sys.argv[4]
# train_out = sys.argv[5]
# test_out = sys.argv[6]
# metrics_out = sys.argv[7]
# num_epoch = sys.argv[8]

formatted_train_input = "model1_formatted_train.tsv"
formatted_validation_input = "model1_formatted_valid.tsv"
formatted_test_input = "model1_formatted_test.tsv"
dict_input = "dict.txt"
train_out = "train_out.labels"
test_out = "test_out.labels"
metrics_out = "metrics_out.txt"
num_epoch = 200
lrn_rate = 0.1


def dot(X, W):
    return np.dot(X, W)


def sigmo(x):
    return (1 / (1 + math.exp(-x)))


def sparse_dot(T, X):
    product = T[0]
    for k in X:
        product += 1 * T[int(k) + 1]
    return product


# store words into dictionary
with open(dict_input, "r") as document:
    dictionary = {}  # dictionary with feature name and label
    label = []  # table with all label name (index) in dictionary
    i = 0
    for line in document:
        line = line.split()
        if not line:
            continue
        dictionary[line[1]] = line[0]
        label.append(line[1])


# create init_table with read in file and corresponding values
def read_file(filename):
    init_table = []
    with open(filename, "r") as doc:
        for line in doc:
            y = line[0]
            new_line = line[1:].split("\t")[1:]
            data_dic = {}  # initiate dictionary
            for element in new_line:
                ele = element.split(":")
                data_dic[ele[0]] = ele[1]
            sublist = [y, data_dic]
            init_table.append(sublist)
    return init_table


def get_y(data):
    y_list = []
    for line in data:
        y_list.append(int(line[0]))
    return y_list


def single_SGD(line_i, theta, lrn_rate):
    new_line = line_i[1]
    y = int(line_i[0])
    theta_x = sparse_dot(theta, new_line)
    exp_dot = math.exp(theta_x)
    diff = y - exp_dot / (1 + exp_dot)
    # bias update
    theta[0] = theta[0] + float(lrn_rate * (diff * 1.00))
    for element in new_line:
        # new theta=original theta-lrn_rate*(diff*Xi)
        theta[int(element) + 1] = theta[int(element) + 1] + float(lrn_rate * (diff * 1.00))
    return theta


def cal_avg(x, theta):
    obj_fun = 0
    for i in range(len(x)):
        y=x[i][0]
        theta_x = sparse_dot(theta, x[i][1])
        obj_fun += -float(y) * theta_x + math.log(1 + math.exp(theta_x))
    return obj_fun/len(x)


def train(train, init_theta, iter_num, x, log):
    temp = init_theta
    for i in range(1, int(iter_num) + 1):
        x.append(i)
        print(i)
        for line in train:
            temp = single_SGD(line, temp, lrn_rate)
        log_i=cal_avg(train,temp)
        log.append(log_i)
    return temp



x = []
log = []
log2 = []
# # get all results for train data
example1 = read_file(formatted_train_input)
example2 = read_file(formatted_validation_input)
train(example1, [0] * (len(label) + 1), num_epoch, x, log)
x2 = []
train(example2, [0] * (len(label) + 1), num_epoch, x2, log2)
print(x)
print(log)
print(log2)
plt.plot(x, log)
plt.plot(x, log2)
plt.show()

# def predict(theta, data):
#     predict_y = []
#     for line in data:
#         new_line = line[1]
#         tmp = sparse_dot(theta, new_line)
#         result = sigmo(tmp)
#         if result >= 0.5:
#             predict_y.append(1)
#         else:
#             predict_y.append(0)
#     return predict_y
#
#
# def get_error(original_y, predicted_y):
#     error = 0
#     for i in range(0, len(original_y)):
#         if predicted_y[i] != original_y[i]:
#             error += 1
#     print(error)
#     print(len(original_y))
#     print(float(error / len(original_y)))
#     return float(error / len(original_y))
#
#
# # export file into label file
# def export(listname, exportname):
#     file = open(exportname, "w")
#     for line in listname:
#         file.write(str(line) + '\n')
#     file.close()


# # get all results for train data
# example1 = read_file(formatted_train_input)
# final_theta = train(example1, [0] * len(label), num_epoch)
# error_rate1 = get_error(get_y(example1), predict(final_theta, example1))
# export(predict(final_theta, example1), train_out)
#
# # get all results for test data
# example2 = read_file(formatted_test_input)
# error_rate2 = get_error(get_y(example2), predict(final_theta, example2))
# export(predict(final_theta, example2), test_out)
#
# # generate metric out file
# # generate label files for train and test data
# file = open(metrics_out, "w")
# file.write("error(train): " + str(error_rate1) + "\n")
# file.write("error(test): " + str(error_rate2) + "\n")
# file.close()
