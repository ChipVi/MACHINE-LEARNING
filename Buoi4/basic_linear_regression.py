import numpy as np

list2D = [[1,2], [3,4]]

data=np.array(list2D)
# print(data)

def calPredictValue(x, theta):
    return np.dot(x.T, theta)

