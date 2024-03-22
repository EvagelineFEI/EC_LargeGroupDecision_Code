import math
from data_manipulate.dataprocess import load_mat
import numpy as np
from S_R_compute.distance import Distance


class GroupRisk:
    def __init__(self, data, alpha,w_matrix, dimension=None, num=30):
        if dimension is None:
            dimension = [4, 4, 3]
        self.m_first_dimension = dimension[0]
        self.m_second_dimension = dimension[1]
        self.m_third_dimension = dimension[2]
        self.num = 30
        self.data = data
        self.w_matrix = w_matrix
        self.matrix_vgij = []
        self.mcov = np.zeros((num, num))
        self.alle = []
        self.count_dis = Distance(data=data, w_matrix=w_matrix,alpha=alpha,dimension=dimension,num=num)

    def count_var2t(self):
        sum = 0
        # for sub0 in range(2*t+1):
        #   sum += t**2
        # return sum/(2*t+1)
        sub1 = 2 * self.m_second_dimension / 2
        for sub0 in range(2 * self.m_second_dimension + 1):
            sum += (sub0 - sub1) ** 2
        return sum / (2 * self.m_second_dimension)

    def count_vgij(self):  # 计算matrix_vgij
        for iter in range(self.num):
            Var2t = 0  # !!!可能是这里导致了R过大
            proc = []
            for i in range(self.m_second_dimension):
                m_gbij = np.sum(self.data[iter][i], axis=1)
                proc.append(m_gbij)
            proc = np.array(proc)
            proc /= self.m_third_dimension

            matrix_vgij_ = np.zeros((self.m_first_dimension, self.m_first_dimension))
            for i in range(self.m_first_dimension):
                for j in range(self.m_first_dimension):
                    inlines = 0
                    for l in range(self.m_third_dimension):
                        inlines += (self.data[iter][i][j][l] - proc[i][j]) ** 2
                        Var2t = self.count_var2t()

                    inlines = inlines / (self.m_third_dimension * Var2t)
                    matrix_vgij_[i][j] = proc[i][j] - inlines

            self.matrix_vgij.append(matrix_vgij_)

    def count_vig(self, matrix_vgij):
        for i in range(self.m_first_dimension):
            matrix_vgij[i][i] = 0
        matrix_vig = np.sum(abs(matrix_vgij), axis=1)
        matrix_vig = np.array(matrix_vig)
        return matrix_vig

    def countP(self, matrix_vig):
        below_sum = np.sum(matrix_vig)
        p = []
        for i in range(self.m_first_dimension):
            p.append(matrix_vig[i] / below_sum)
        p = np.array(p)
        return p

    def count_var(self, e, matrix_vig):  # 这是计算单个值（每个决策矩阵的方差），最后要放入数组
        inlinesum = 0
        for i in range(self.m_first_dimension):
            inlinesum += (matrix_vig[i] - e[i]) ** 2
        return inlinesum / self.m_first_dimension

    def count_alle(self):
        allp = []
        for i in range(self.num):
            m0 = self.matrix_vgij[i]
            m_vig = self.count_vig(m0)
            allp.append(self.countP(m_vig))

        for i in range(self.num):
            m0 = self.matrix_vgij[i]
            m_vig = self.count_vig(m0)
            p0 = allp[i]
            self.alle.append(np.multiply(m_vig, p0))

    def count_mcov(self):  # 计算num*num矩阵 self.mcov
        def count_cov_(m_vig, m_vik, eg, ek):
            m_vig -= eg
            m_vik -= ek
            mul = np.multiply(m_vig, m_vik)
            cov = 0
            for i in range(self.m_first_dimension):
                cov += mul[i] / self.m_first_dimension
            return cov

        # allvar = []
        # for i in range(self.num):
        #     m_vig = self.count_vig(self.matrix_vgij[i])
        #     allvar.append(self.count_var(alle[i], m_vig))

        for i in range(self.num):
            for j in range(self.num):
                m_vig = self.count_vig(self.matrix_vgij[i])
                m_vik = self.count_vig(self.matrix_vgij[j])
                self.mcov[i][j] = count_cov_(m_vig, m_vik, self.alle[i], self.alle[j])

    def count_r(self, mw, mvar, mcov):
        add1 = 0
        m_wvar = np.multiply(self.w_matrix, mvar)
        for g in range(self.num):
            add1 += m_wvar[g] ** 2

        add2 = 0
        for g in range(self.num):
            for k in range(self.num):
                if k != g:
                    add2 += mw[g] * mw[k] * mcov[g][k] * mvar[g] * mvar[k]
        add3 = 0
        s, s_list = self.count_dis.final_count()
        s_list = np.array(s_list)
        w_s = np.multiply(self.w_matrix, s_list)
        # print("w_s-------------",w_s)
        for g in range(self.num):
            add3 += w_s[g]
        # print("add3-------------",add3)
        return add1 + add2 + add3

    def final_count(self):
        self.count_vgij()
        self.count_alle()
        allvar = []
        for i in range(self.num):
            m_vig = self.count_vig(self.matrix_vgij[i])
            allvar.append(self.count_var(self.alle[i], m_vig))

        self.count_mcov()
        r_group = self.count_r(self.w_matrix, allvar, self.mcov)

        return r_group
