import numpy as np
import sys

test_input = "toytest.txt"
index_to_word = "toy_index_to_word.txt"
index_to_tag = "toy_index_to_tag.txt"
hmmprior = "hmmprior.txt"
hmmemit = "hmmemit.txt"
hmmtrans = "hmmtrans.txt"
predicted_file = "predicted.txt"
metric_file = "metrics.txt"


## store data tag and word into dictionary
def getDictionary(filename):
    with open(filename, "r") as doc:
        i = 0  # index start from 0
        dictionary = {}
        con_dictionary = {}
        for line in doc:
            line = line.rstrip()
            dictionary[line] = i
            con_dictionary[i] = line
            i = i + 1
        return [dictionary, con_dictionary]


# read test data as test_input
## read train file as input
def read_test(filename):
    with open(filename, "r") as doc:
        file = []
        for line in doc:  # for each line
            subline = []
            newline = line.rstrip()
            word = newline.split(" ")  # store each word in the line
            for word in word:  # for each word element
                text = word.split("_")[0]
                tag = word.split("_")[1]
                element = [text, tag]
                subline.append(element)
            file.append(subline)
        return file


def read_param(filename):
    with open(filename, "r") as doc:
        file = []
        for line in doc:
            newline = line.rstrip()
            prob = newline.split(" ")
            file.append(prob)
        file = np.array(file)
        return file


# forward process
# remember to put normalize
def forward(line, worddic, tagdic, prior, trans, emit):
    alpha = [[0] * len(tagdic)] * len(line)
    index_1 = worddic[line[0][0]]
    alpha[0] = np.multiply(prior, np.array(emit[index_1], dtype=float))
    for i in range(1, len(line)):
        index_i = worddic[line[i][0]]
        # alpha[i - 1] = normalize(alpha[i - 1])
        alpha[i] = np.dot(alpha[i - 1], trans) * emit[index_i]
    return alpha


# backward process
def backward(line, worddic, tagdic, trans, emit):
    beta = [[0] * len(tagdic)] * len(line)
    beta[len(line) - 1] = [1] * len(tagdic)
    for i in range(len(line) - 2, -1, -1):
        index_i = worddic[line[i + 1][0]]
        beta[i] = np.dot(trans, beta[i + 1] * emit[index_i])
        # beta[i] = normalize(beta[i])
    return beta


def normalize(line):
    SUM = sum(line)
    result = []
    for ele in line:
        result.append(ele / SUM)
    return result


def run(testfile):
    log_sum = []
    prediction = []
    for line in testfile:
        alpha = forward(line, word_dic, tag_dic, prior, trans, emit)
        print(alpha)
        beta = backward(line, word_dic, tag_dic, trans, emit)
        print(beta)
        # get the log-likelihood
        LLH = np.log(sum(alpha[len(line) - 1]))
        log_sum.append(LLH)
        # get the tag probability
        prob = np.array(alpha) * beta
        # every element represent the prob of different tag
        sub_prediction = []  # the prediction of this line
        for element in prob:
            index_max = np.argmax(element)
            sub_prediction.append(index_max)
        prediction.append(sub_prediction)
    log_likelihood = np.average(log_sum)
    return [log_likelihood, prediction]


def get_prediction(prediction_index, con_tag_dic, testfile):
    prediction = ""
    l = 0
    error = 0
    total_sum = 0
    for line in prediction_index:
        i = 0
        pred_line = ""
        for ele in line:
            text = testfile[l][i][0]
            tag = con_tag_dic[ele]
            if (testfile[l][i][1] != tag):
                error = error + 1  # get the error
            i = i + 1  # get index in the line, also calculate number of element
            pred_word = text + "_" + tag
            pred_line = pred_line + pred_word + " "
        total_sum = total_sum + i
        pred_line = pred_line.rstrip()
        prediction = prediction + pred_line + "\n"
        l = l + 1
    accuracy = (total_sum - error) / total_sum
    prediction = prediction.rstrip()
    return [prediction, accuracy]


# read index file as dictionary
tag_dic = getDictionary(index_to_tag)[0]
con_tag_dic = getDictionary(index_to_tag)[1]
word_dic = getDictionary(index_to_word)[0]
con_word_dic = getDictionary(index_to_word)[1]
test_file = read_test(test_input)

# read prior
with open(hmmprior, "r") as doc:
    prior = []
    for line in doc:
        prior.append(line)
prior = np.array(prior, dtype=float)

# read hmmemit
emit = read_param(hmmemit)
emit = np.transpose(np.array(emit, dtype=float))

# read hmmtrans
trans = read_param(hmmtrans)
trans = np.array(trans, dtype=float)

# run forward and backward algorithm for log-likelihood and prediction
run_result = run(test_file)
log_likelihood = run_result[0]
prediction_index = run_result[1]
print(log_likelihood)
prediction = get_prediction(prediction_index, con_tag_dic, test_file)[0]
print(prediction)
accuracy = get_prediction(prediction_index, con_tag_dic, test_file)[1]
# print(prediction)
print(accuracy)

# export prediction and metric file
file = open(predicted_file, "w")
file.write(prediction)
file.close()

file = open(metric_file, "w")
file.write("Average Log-Likelihood: " + str(log_likelihood) + "\n")
file.write("Accuracy: " + str(accuracy))
file.close()
