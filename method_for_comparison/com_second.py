#  rij就用矩阵中的元素;upper和down就用最大和最小值
# compared method 2
import numpy as np
class IntervalTypeFuzzy:
    def __init__(self, data, alpha, theta=2):
        self.data = data
        self.theta = theta
        self.alpha = alpha
        self.fuz = 0
        self.hes = 0
        self.first_dimension = len(self.data) # m
        self.second_dimension = len(self.data[0]) # n

    def count_fuzz(self):
        shape = np.array(self.data[0][0]).shape
        for i in range(self.first_dimension):
            for j in range(self.second_dimension):
                #矩阵 self.data[i][j]
                up_sum= 0
                for k in range(shape[0]):
                    up_sum += (max(self.data[i][j][k])-min(self.data[i][j][k]))
                low_sum = 0
                for k in range(shape[0]):
                    low_sum += max(self.data[i][j][k])
                self.fuz += up_sum/low_sum

        self.fuz = self.fuz/(self.first_dimension*self.second_dimension)

    def count_hes(self):
        shape = np.array(self.data[0][0]).shape
        for i in range(self.first_dimension):
            for j in range(self.second_dimension):
                # 矩阵 self.data[i][j]
                up_sum = 0
                for k in range(shape[0]):
                    up_sum += 2 * min(self.data[i][j][k])
                low_sum = 0
                for k in range(shape[0]):
                    low_sum += max(self.data[i][j][k])
                self.hes += up_sum / low_sum

        self.hes = self.hes / (self.first_dimension * self.second_dimension)

    def count_r(self):
        self.count_fuzz()
        self.count_hes()
        return ((2**self.fuz - 1)**self.alpha + (2**self.hes - 1)**self.alpha)/self.theta

