import csv
import math
import sys
import numpy as np

with open(sys.argv[1], "r") as csvfile:
    csvReader = csv.reader(csvfile, delimiter=',')
    data = [data for data in csvReader]
    data_array = np.asarray(data)
    col_length = len(data_array[0, :])


    ##calculate error rate
    def get_error(list):
        counts_elements = np.unique(list, return_counts=True)
        element_array = np.asarray(counts_elements)[1, :]
        element_array = np.asarray(element_array, dtype=int)
        error_rate = min(element_array) / sum(element_array)
        return error_rate


    ##calculate entropy
    def get_entropy(list):
        p = get_error(list[:, col_length - 1])
        if p == 0 or p == 1:
            entropy = 0
        else:
            entropy = p * (-math.log(p, 2)) + (1 - p) * (-math.log((1 - p), 2))
        return entropy

data_array = np.delete(data_array, 0, 0)
final_error = get_error(data_array[:, col_length - 1])
final_entropy = get_entropy(data_array)
file = open(sys.argv[2], "w+")
L = ["entropy: ", str(final_entropy), "\n", "error: ", str(final_error), "\n"]
file.writelines(L)

file.close
