import numpy as np

#### forward####
x = np.array([1, 1, 0, 0, 1, 1])
#a = np.array([[1, 2, -3, 0, 1, -3], [3, 1, 2, 1, 0, 2], [2, 2, 2, 2, 2, 1], [1, 0, 2, 1, -2, 2]])
a = np.array([[10, 20, -30, 0, 10, -30], [30, 10, 20, 10, 0, 20], [20, 20, 20, 20, 20, 10], [10, 0, 20, 10, -20, 20]])
b = np.array([[1, 2, -2, 1], [1, -1, 1, 2], [3, 1, -1, 1]])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


z0 = 1
a1 = np.dot(x, a[0]) + 1
z1 = sigmoid(a1)
print("a1: " + str(a1) + " z1: " + str(z1))

a2 = np.dot(x, a[1]) + 1
z2 = sigmoid(a2)
print("a2: " + str(a2) + " z2: " + str(z2))

a3 = np.dot(x, a[2]) + 1
z3 = sigmoid(a3)
print("a3: " + str(a3) + " z3: " + str(z3))

a4 = np.dot(x, a[3]) + 1
z4 = sigmoid(a4)
print("a4: " + str(a4) + " z4: " + str(z4))

z = [z1, z2, z3, z4]
b1 = np.dot(z, b[0]) + 1
b2 = np.dot(z, b[1]) + 1
b3 = np.dot(z, b[2]) + 1
beta = [b1, b2, b3]
print(beta)

y1 = np.exp(b1) / (np.exp(b1) + np.exp(b2) + np.exp(b3))
y2 = np.exp(b2) / (np.exp(b1) + np.exp(b2) + np.exp(b3))
y3 = np.exp(b3) / (np.exp(b1) + np.exp(b2) + np.exp(b3))
y_pre = [y1, y2, y3]
print(y_pre)

weight = 0
for element in a:
    for x in element:
        weight += np.square(x)
weight = np.sqrt(weight)

y = [0, 1, 0]
loss = 0
for i in range(0, 3):
    tmp = y[i] * np.log(y_pre[i])
    loss += tmp
loss = -loss
reg_loss = loss + 0.01 * weight / (2 * 24)
print("loss is: " + str(loss))
print("Regulized loss is: " + str(reg_loss))

### backpropogation ###
theta21 = -(1 - y2) * (y2 * (1 - y2)) * z1
print("updated value of b21: " + str(1 - theta21))
bias = -(0 - y1) * (y1 * (1 - y1)) * z0
print("updated weight of hidden layer: " + str(1 - bias))
beta33 = 1 - (-(0 - y3) * (y3 * (1 - y3)) * z3)
alpha34 = 0 * z3 * (1 - z3) * (-y3) * y3 * (1 - y3)
print("updated value alpha34: " + str(2 - alpha34))
alpha20 = 1 * z2 * (1 - z2) * (-1) * (-(1 - y2) * (y2 * (1 - y2)))
print("updated alpha20 is: " + str(1 - alpha20))