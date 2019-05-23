import numpy as np
import sys
import math
import matplotlib.pyplot as plt

# train_input = sys.argv[1]
# test_input = sys.argv[2]
# train_out = sys.argv[3]
# test_out = sys.argv[4]
# metrics_out = sys.argv[5]
# num_epoch = sys.argv[6]
# hidden_units = sys.argv[7]
# init_flag = sys.argv[8]
# learning_rate=sys.argv[9]

train_input = "largeTrain.csv"
test_input = "largeTest.csv"
train_out = "modeltrain_out.labels"
test_out = "modeltest_out.labels"
metrics_out = "modelmetrics_out.labels"
num_epoch = "100"
hidden_units = "50"
init_flag = "1"
#learning_rate = "0.01"
K = 10
M = 128


class Object:
    def __init__(self, x, a, z, b, y_hat, J):
        self.x = x
        self.a = a
        self.z = z
        self.b = b
        self.y_hat = y_hat
        self.J = J


def dot(X, W):
    return np.array(np.dot(X, W))


# read file and store in a list, each line contains an array with all x and y
def read_file(filename):
    data = []
    with open(filename, "r") as doc:
        for line in doc:
            y = float(line[0])
            new_line = line[1:].split(",")
            new_line = new_line[1:]
            subarray = [1]  # add a column as one in every line of X
            for ele in new_line:
                subarray.append(float(ele.rstrip()))
            subarray = np.array(subarray)
            subline = [y, subarray]
            data.append(subline)
    return data


# get y from the original data
def getY(filename):
    y = []
    with open(filename, "r") as doc:
        for line in doc:
            y.append(float(line[0]))
    return y


def init_weight(flag, K, D, M):
    alpha = []
    beta = []
    if flag == "1":
        for d in range(0, int(D)):
            a = [0.00]  # initate bias as zero
            for m in range(0, M):
                random = float(np.random.uniform(-0.1, 0.1, 1))
                a.append(random)
            alpha.append(a)
        for k in range(0, K):
            b = [0.00]  # initiate bias as zero
            for d in range(0, int(D)):
                random = float(np.random.uniform(-0.1, 0.1, 1))
                b.append(random)
            beta.append(b)
    else:
        a = [0] * (M + 1)
        for d in range(0, int(D)):
            alpha.append(a)
        b = [0] * (int(D) + 1)
        for k in range(0, K):
            beta.append(b)
    alpha = np.array(alpha)
    beta = np.array(beta)
    result = [alpha, beta]
    return result


# sigmoid every element in vector x in forward process
def sigmoidForward(x):
    z = [1]  # add one column as 1 in every z
    for ele in x:
        result = 1 / (1 + np.exp(-ele))
        z.append(result)
    z = np.array(z)
    return z


def softmaxForward(b):
    sum = 0
    y_hat = []
    for ele in b:
        sum += np.exp(ele)
    for bk in b:
        result = np.exp(bk) / sum
        y_hat.append(result)
    y_hat = np.array(y_hat)
    return y_hat


def NNForward(x, y, alpha, beta):
    a = dot(alpha, x)  # a is a vector of first layer pre-result, alpha()
    z = sigmoidForward(a)  # z sigmoid every element in a, 1st layer result
    b = dot(beta, z)  # b is a vector of second layer pre-result
    y_hat = softmaxForward(b)  # y is the probability based on b, a k dimention vector
    J = -np.log(y_hat[int(y)])  # y is a one-hot vector, we can regardless y_predict on index that is not y
    o = Object(x, a, z, b, y_hat, J)
    return o


def NNBackward(x, y, alpha, beta, o, learning_rate):
    # 6
    y_array = np.array([0] * K)
    y_array[int(y)] = 1
    g_yhat = np.array([0] * K)
    g_yhat[int(y)] = -1 / o.y_hat[int(y)]
    # 7 1*K
    diag = np.diag(o.y_hat)
    y_hat = o.y_hat[:, None]
    # gb = np.dot(np.transpose(g_yhat), (diag - np.dot(y_hat, np.transpose(y_hat))))
    gb = o.y_hat - y_array
    z = np.array(o.z)
    z = z[:, None]
    # 8 10*5
    gb = gb[:, None]
    g_beta = np.dot(gb, np.transpose(z))
    gz = np.dot(np.transpose(beta), gb)
    gz = np.delete(gz, 0, 0)  # 4*1
    # 10
    z = z[1:]  # remove the column with value 1
    ONE = np.array([1] * int(hidden_units))
    ONE = ONE[:, None]
    ga = np.array(np.multiply(np.multiply(gz, z), np.subtract(ONE, z)))
    # 11
    x = np.array(x[:, None])
    g_alpha = dot(ga, np.transpose(x))
    return [g_alpha, g_beta]


def SGD_train(train_data, test_data, learning_rate):
    init_theta = init_weight(init_flag, K, hidden_units, M)
    alpha = init_theta[0]
    beta = init_theta[1]
    J_train = []
    J_test = []
    for i in range(1, int(num_epoch) + 1):
        print(i)
        for line in train_data:
            # compute neural network layers
            o = NNForward(line[1], line[0], alpha, beta)
            # compute gradient via backprop
            gradient = NNBackward(line[1], line[0], alpha, beta, o, hidden_units)
            g_alpha = gradient[0]
            g_beta = gradient[1]
            g_alpha[:] = [g * float(learning_rate) for g in g_alpha]
            g_beta[:] = [g * float(learning_rate) for g in g_beta]
            # update parameters
            alpha = np.subtract(alpha, g_alpha)
            beta = np.subtract(beta, g_beta)
        J_train.append(predict(train_data, alpha, beta)[1])
        J_test.append(predict(test_data, alpha, beta)[1])
    print(J_train)
    print(J_test)
    return [alpha, beta, J_train, J_test]


def predict(data, alpha, beta):
    label = []
    entropy = 0
    for line in data:
        y = line[0]
        o = NNForward(line[1], y, alpha, beta)
        label.append(np.argmax(o.y_hat))
        entropy += o.J
    combine = [label, entropy / len(data)]
    return combine


def errorRate(original_y, predict_y):
    error = 0
    for i in range(0, len(original_y)):
        if original_y[i] != predict_y[i]:
            error += 1
    print(error / len(original_y))
    return error / len(original_y)


# export file into label file
def export(listname, exportname):
    file = open(exportname, "w")
    for line in listname:
        file.write(str(line) + '\n')
    file.close()


# def exportEntropy(trainlist, testlist, exportname):
#     file = open(exportname, "w")
#     for i in range(0, int(num_epoch)):
#         train = trainlist[i]
#         test = testlist[i]
#         file.write("epoch=" + str(i + 1) + " crossentropy(train): " + str(train) + '\n')
#         file.write("epoch=" + str(i + 1) + " crossentropy(test): " + str(test) + '\n')
#     # error generated
#     file.write("error(train): " + str(train_error) + '\n')
#     file.write("error(test): " + str(test_error) + '\n')


train_data = read_file(train_input)
test_data = read_file(test_input)

# train_y = getY(train_input)
# test_y = getY(test_input)
#
# theta = SGD_train(train_data, test_data)
# alpha = theta[0]
# beta = theta[1]
# trainEntropy = theta[2]
# testEntropy = theta[3]
#
# train_label = predict(train_data, alpha, beta)[0]
# test_label = predict(test_data, alpha, beta)[0]
#
# train_error = errorRate(train_y, train_label)
# test_error = errorRate(test_y, test_label)

# export label
# export(train_label, train_out)
# export(test_label, test_out)
#
# print(train_label)
# print(test_label)

# export metrics include error, entropy
# exportEntropy(trainEntropy, testEntropy, metrics_out)

## generate figure in write up
# x = [5, 20, 50, 100, 200]
x = [0.1, 0.01, 0.001]


# def generateGraph(train, test):
#     train_entropy = []
#     test_entropy = []
#     for units in x:
#         theta = SGD_train(train_data, test_data, units)
#         trainresult = predict(train, theta[0], theta[1])
#         testresult = predict(test, theta[0], theta[1])
#         train_entropy.append(trainresult[1])
#         test_entropy.append(testresult[1])
#     return [train_entropy, test_entropy]

def generateGraph(train, test):
    train_entropy = []
    test_entropy = []
    for lrn_rate in x:
        theta = SGD_train(train_data, test_data, lrn_rate)
        trainresult = predict(train, theta[0], theta[1])
        testresult = predict(test, theta[0], theta[1])
        train_entropy.append(trainresult[1])
        test_entropy.append(testresult[1])
    print(train_entropy)
    print(test_entropy)
    return [train_entropy, test_entropy]


# plotresult = generateGraph(train_data, test_data)

dependent1=[0.02189110967400229, 0.05291396198727231, 0.3759657804654571]
dependent2=[0.8304738700315272, 0.44949633956306156, 0.5168232660201509]

# plt.plot(x, plotresult[0], label='train')
# plt.plot(x, plotresult[1], label='test')
plt.plot(x, dependent1, label='train')
plt.plot(x, dependent2, label='test')
plt.xlim(0.1, 0)
plt.xlabel("learning rate")
plt.ylabel("Average Cross Entropy")
plt.legend()
plt.show()
