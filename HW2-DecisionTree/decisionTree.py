import numpy as np
import math
import sys
import csv

train_input = sys.argv[1]
test_input = sys.argv[2]
max_depth = int(sys.argv[3])
train_out = sys.argv[4]
test_out = sys.argv[5]
metrics_out = sys.argv[6]


# reading file
def readfile(data):
    with open(data, "r") as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',')
        data = [data for data in csvReader]
        data_array = np.asarray(data)
        return data_array


####calculate error rate####
def get_error_rate(data):
    counts_elements = np.unique(data, return_counts=True)
    element_array = np.asarray((counts_elements))[1, :]
    element_array = np.asarray(element_array, dtype=int)
    error_rate = min(element_array) / sum(element_array)
    return error_rate


### get most frequent value in column
def get_max_value(data):
    counts_elements = np.unique(data, return_counts=True)
    value_element = np.asarray(counts_elements)[0, :]
    element_array = np.asarray(counts_elements)[1, :]
    element_array = np.asarray(element_array, dtype=int)
    length1 = len(element_array)
    if length1 == 1:
        value = value_element[0]
        value2 = " "
        count = element_array[0]
        count2 = 0
    elif element_array[0] >= element_array[1]:
        value = value_element[0]
        value2 = value_element[1]
        count = element_array[0]
        count2 = element_array[1]
    else:
        value = value_element[1]
        value2 = value_element[0]
        count = element_array[1]
        count2 = element_array[0]
    return [value, value2, count, count2]


### get entropy for attribute at data[index]
def get_sub_entropy(data, index):
    single_list = data[:, index]
    counts_elements = np.unique(single_list, return_counts=True)
    value_element = np.asarray(counts_elements)[0, :]
    element_array = np.asarray(counts_elements)[1, :]
    element_array = np.asarray(element_array, dtype=int)

    ##get sub entropy for first value in y
    new_subset1 = data[data[:, index] == value_element[0]]
    length = len(new_subset1[0, :])
    p_y1 = get_error_rate(new_subset1[:, length - 1])
    if p_y1 == 0 or p_y1 == 1:
        entropy1 = 0
    else:
        entropy1 = p_y1 * (-math.log(p_y1, 2)) + (1 - p_y1) * (-math.log((1 - p_y1), 2))
    p1 = element_array[0] / sum(element_array)

    ##only get sub entropy for second value in y when there are two values in y
    if len(value_element) > 1:
        new_subset2 = data[data[:, index] == value_element[1]]
        length = len(new_subset2[0, :])
        p_y2 = get_error_rate(new_subset2[:, length - 1])
        if p_y2 == 0 or p_y2 == 1:
            entropy2 = 0
        else:
            entropy2 = p_y2 * (-math.log(p_y2, 2)) + (1 - p_y2) * (-math.log((1 - p_y2), 2))
        p2 = element_array[1] / sum(element_array)
    else:
        p2 = 0
        entropy2 = 0

    entropy = p1 * entropy1 + p2 * entropy2
    return entropy


### get multual information for each attributes
def get_mutual_info(data, col_length):
    mutual_info = []
    # get entropy for Y
    y_p = get_error_rate(data[:, col_length - 1])
    if y_p == 0 or y_p == 1:
        y_entropy = 0
    else:
        y_entropy = y_p * (-math.log(y_p, 2)) + (1 - y_p) * (-math.log((1 - y_p), 2))

    # get entropy for different attributes
    for i in range(0, col_length - 1):
        entropy = get_sub_entropy(data, i)
        x_info_tmp = y_entropy - entropy
        if x_info_tmp > 0:
            mutual_info.append(x_info_tmp)
        else:
            mutual_info.append(0)
    return mutual_info


class DecisionNode():
    def __init__(self, value=None, aplit_attr=None, left_branch=None, left_value=None,
                 right_branch=None, right_value=None, leaf=None, left_common=None, right_common=None):
        self.value = value  # the value got from last node
        self.split_attr = aplit_attr
        self.left_branch = left_branch
        self.right_branch = right_branch
        self.left_value = left_value
        self.right_value = right_value
        self.leaf = leaf
        self.left_common = left_common
        self.right_common = right_common


def get_tree(data, max_depth, depth, error_rate, y_max, attribute, assign_value):
    root_node = DecisionNode()
    col_length = len(data[0, :])
    if max_depth == 0:
        root_node.leaf = get_max_value(data[:, col_length - 1])[0]
    else:
        # conditions that the recursive will stop
        if depth > max_depth or error_rate == 0 or error_rate == 1:
            root_node.leaf = y_max[0]
            root_node.value = assign_value
        else:
            # get max mutual inform for data attributes
            mutual_info = get_mutual_info(data, col_length)
            index = mutual_info.index(max(mutual_info))
            attribute_name = attribute[index]

            # get unique value of column data[:,index]
            elements = np.unique(data[:, index], return_counts=False)
            element_array = np.asarray(elements)
            attribute = np.delete(attribute, index, axis=0)

            # generate left and right new data
            if len(element_array) > 1:
                data_left = data[data[:, index] == element_array[0]]
                data_right = data[data[:, index] == element_array[1]]

                y_left = data_left[:, len(data_left[0, :]) - 1]
                y_right = data_right[:, len(data_left[0, :]) - 1]

                error_y = get_error_rate(y_left)
                error_r = get_error_rate(y_right)

                max_y_left = get_max_value(y_left)
                max_y_right = get_max_value(y_right)

                data_left = np.delete(data_left, index, axis=1)
                data_right = np.delete(data_right, index, axis=1)

                # left and right branch
                root_node.left_branch = get_tree(data_left, max_depth, depth + 1, error_y, max_y_left, attribute,
                                                 element_array[0])
                root_node.right_branch = get_tree(data_right, max_depth, depth + 1, error_r, max_y_right, attribute,
                                                  element_array[1])
                root_node.right_value = element_array[1]
                root_node.right_common = max_y_right
            else:
                data_left = data[data[:, index] == element_array[0]]
                y_left = np.unique(data_left[:, len(data_left[0, :]) - 1])
                error_y = get_error_rate(y_left)
                max_y_left = get_max_value(y_left)
                root_node.left_branch = get_tree(data_left, max_depth, depth + 1, error_y, max_y_left, attribute,
                                                 element_array[0])

            root_node.split_attr = attribute_name
            root_node.value = assign_value
            root_node.left_value = element_array[0]
            root_node.left_common = max_y_left

    return root_node


def print_tree(tree, max_depth, depth):
    if tree.left_branch is not None or tree.right_branch is not None:
        print(
            depth * "|" + tree.split_attr + " = " + str(tree.left_value) + ": [" + str(tree.left_common[2]) + " " + str(
                tree.left_common[0]) + " / " + str(tree.left_common[3]) + " " + str(tree.left_common[1]) + "]")
        if tree.left_branch is not None:
            print_tree(tree.left_branch, max_depth, depth + 1)
        print(depth * "|" + tree.split_attr + " = " + str(tree.right_value) + ": [" + str(
            tree.right_common[2]) + " " + str(tree.right_common[0]) + " / " + str(
            tree.right_common[3]) + " " + str(tree.right_common[1]) + "]")
        if tree.right_branch is not None:
            print_tree(tree.right_branch, max_depth, depth + 1)


def prediction(data, pre_attribute, tree):
    while tree is not None:
        if tree.leaf is not None and tree.left_branch is None and tree.right_branch is None:
            return tree.leaf
            break
        else:
            attribute_str = str(tree.split_attr).strip()
            index = pre_attribute.index(attribute_str)
            if data[index] == str(tree.left_value):
                tree = tree.left_branch
                prediction(data, pre_attribute, tree)
            else:
                tree = tree.right_branch
                prediction(data, pre_attribute, tree)


def get_predict_output(test_input, test_out):
    # get predict data stored in the list without space in the word
    predict_data = readfile(test_input)
    predict_attribute = predict_data[0, :]
    predict_attribute_list = []
    for e in predict_attribute:
        j = e.replace(' ', '')
        predict_attribute_list.append(j)
    predict_data = np.delete(predict_data, 0, 0)

    # store the predict value in a list
    leaf_summary = []

    # get prediction and output file
    file = open(test_out, "w")
    for row in predict_data:
        leaf = prediction(row, predict_attribute_list, root_node)
        leaf_summary.append(leaf)
        file.write(str(leaf) + "\n")
    file.close()
    return leaf_summary


def final_error_rate(test_input, test_out):
    predict_list = get_predict_output(test_input, test_out)
    predict_length = len(predict_list)

    original_data = readfile(test_input)
    length = len(original_data[0, :])
    original_data = np.delete(original_data, 0, 0)
    original_list = original_data[:, length - 1]
    original_length = len(original_list)

    count = 0
    for i in range(0, original_length):
        if predict_list[i] != original_list[i]:
            count = count + 1
    error_rate = count / predict_length
    return error_rate


# read train tree data
train_data = readfile(train_input)
pure_data = np.delete(train_data, 0, 0)

# get attribute element
attribute = train_data[0, :]

# handle situation when max_depth larger than attribute number
if max_depth > len(attribute) - 1:
    max_depth = len(attribute) - 1

# generate tree with root node
length = len(pure_data[0, :])
root_node = get_tree(pure_data, max_depth, 1, get_error_rate(pure_data[:, length - 1]),
                     get_max_value(pure_data[:, length - 1]), attribute, None)

# print tree
print_tree(root_node, max_depth, 1)

# generate metric out file
# generate label files for train and test data
file = open(metrics_out, "w")
file.write("error(train): " + str(final_error_rate(train_input, train_out)) + "\n")
file.write("error(test): " + str(final_error_rate(test_input, test_out)) + "\n")
file.close()
