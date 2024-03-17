import math
import numpy as np


class Distance:

    def __init__(self, data, w_matrix, num=30):
        self.n = 4
        self.L = 3
        self.t = 4
        self.num = num
        self.data = data
        self.w_matrix = w_matrix
    def num_setter(self):
        self.num = len(self.w_matrix)

    def inside_proc(self, a):
        return (np.abs(a) / (2 * self.t)) ** 2

    def multi_add(self, m1, m2):
        matrix_sub = m1 - m2  # 2个矩阵对应位置上的元素相减
        proc = self.inside_proc(matrix_sub)
        add_all = np.sum(proc)
        return add_all

    def count_similar(self, add_all):
        return 1 - math.sqrt(1 / (self.L * (self.n ** 2)) * add_all)

    def final_count(self):
        store_dis1 = np.zeros((self.num, self.num))
        for o1 in range(self.num - 1):  # 0~32
            matrix_1 = self.data[o1]
            for o2 in range(o1 + 1, self.num):  # 边界上可能有bug
                matrix_2 = self.data[o2]
                addall = self.multi_add(matrix_1, matrix_2)
                siml = self.count_similar(addall)
                store_dis1[o1][o2] = siml
        bf = np.zeros((4, 4, 3))
        for o1 in range(self.num):
            matrix = self.data[o1]
            bf += matrix * self.w_matrix[o1]

        sgroup = 0
        s_list = []
        for o1 in range(self.num):
            matrix_g = self.data[o1]
            addall = self.multi_add(matrix_g, bf)
            siml1 = self.count_similar(addall)
            sgroup += siml1
            s_list.append(siml1)
        # time_end = time.time()
        # time_c = time_end - time_start
        # print("Time cost in calculateDistance is:",time_c)
        return sgroup, s_list