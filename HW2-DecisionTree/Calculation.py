import numpy as np

# X1 = np.matrix([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
# X2 = np.matrix([[1, 1, 1, 1, 1], [1, 2, 3, 4, 5]])
#
# X3 = np.matrix(np.dot(X1, X2))
#
# print(X3)


X1 = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
X2 = np.array([[1, 1, 1, 1, 1], [1, 2, 3, 4, 5]])
Y=np.array([3,8,9,12,15])
X3 = np.dot(X2, X1)
X4 = np.array(X3)
X5 = np.linalg.inv(X4)

X6 = np.dot(X5, X2)
X7=np.dot(X6,Y)
print(X7)
