import sys

train_input = sys.argv[1]
validation_input = sys.argv[2]
test_input = sys.argv[3]
dict_input = sys.argv[4]
formatted_train_out = sys.argv[5]
formatted_validation_out = sys.argv[6]
formatted_test_out = sys.argv[7]
feature_flag = sys.argv[8]

# train_input = "smalltrain_data.tsv"
# validation_input = "smallvalid_data.tsv"
# test_input = "smalltest_data.tsv"
# dict_input = "dict.txt"
# formatted_train_out = "formatted_train.tsv"
# formatted_validation_out = "formatted_validate.tsv"
# formatted_test_out = "formatted_test.tsv"
# feature_flag = 1

# store words into dictionary
with open(dict_input, "r") as document:
    dictionary = {}
    for line in document:
        line = line.split()
        if not line:
            continue
        dictionary[line[0]] = line[1]


# read file
def read_file(filename):
    list = []
    with open(filename, "r") as doc:
        for line in doc:
            sublist = line.split()
            list.append(sublist)
    return list


def wordCount(listname):
    list = []
    for line in listname:
        new_line = line[1:]
        uniqWords = sorted(set(new_line))  # remove duplicate words and sort
        sublist = [line[0]]
        for word in uniqWords:
            sublist.append([word, new_line.count(word)])
        list.append(sublist)
    return list


def format(listname, featureflag):
    list = []
    for line in listname:
        new_line = line[1:]
        sublist = [line[0]]
        for x in new_line:
            if x[0] in dictionary:
                if featureflag == '1':
                    sublist.append(dictionary[x[0]] + ":1")
                else:
                    if x[1] < 4:
                        sublist.append(dictionary[x[0]] + ":1")
        list.append(sublist)
    return list


def export(listname, exportname):
    file = open(exportname, "w")
    for line in listname:
        string = '\t'.join(line)
        file.write(string + '\n')
    file.close()


# export train data
export(format(wordCount(read_file(train_input)), feature_flag), formatted_train_out)

# export test data
export(format(wordCount(read_file(test_input)), feature_flag), formatted_test_out)

# export validate data
export(format(wordCount(read_file(validation_input)), feature_flag), formatted_validation_out)
