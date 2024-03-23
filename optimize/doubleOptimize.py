from random import random
from random import sample
from random import uniform
import numpy as np
from S_R_compute.distance import Distance
from S_R_compute.risk import GroupRisk
import json


class DoubleOptimizer:
    def __init__(self, data, alpha,dimension=None, num=30, popsize=100, mutate=0.8, recombination=0.5, maxtier=100, elite=0.8):
        if dimension is None:
            dimension = [4, 4, 3]
        self.popsize = popsize
        self.mutate = mutate
        self.cr = recombination
        self.maxtier = maxtier
        self.elite = elite
        self.population = []
        self.gens_scores = []
        self.forselect_pop = []
        self.gens_avg_record = []
        self.genr_avg_record = []
        self.gen_best_record = [[] for i in range(maxtier)]  # 可视化: 每次迭代后，都显示出S和R的值
        self.gen_solset = [[] for i in range(maxtier)]
        self.num = num
        self.bounds = (0, 1)
        self.w_matrix = np.zeros(num)
        self.distance_counter = Distance(data, self.w_matrix, alpha,dimension,num)
        self.risk_counter = GroupRisk(data, self.w_matrix, alpha,dimension,num)


    def addto1(self,pop):
        for ind in range(len(pop)):
            allsum = np.sum(pop[ind])
            pop[ind] /= allsum

    def pop_init(self):
        for i in range(self.popsize):
            indiv = []
            for j in range(self.num):
                indiv0 = uniform(0, 1)
                if indiv0 != 0:
                    indiv.append(indiv0)
            self.population.append(indiv)
        self.addto1(self.population)

    def dominate(self, p, q):  # p q 是两个向量,计算 p 是否支配 q
        self.distance_counter.w_matrix = p.copy()
        self.risk_counter.w_matrix = p.copy()
        p_1, _ = self.distance_counter.final_count()
        p_2 = self.risk_counter.final_count()

        self.distance_counter.w_matrix = q.copy()
        self.risk_counter.w_matrix = q.copy()
        q_1, _ = self.distance_counter.final_count()
        q_2 = self.risk_counter.final_count()
        global temp_t1
        temp_t1 = p_1
        global temp_t2
        temp_t2 = p_2
        if p_1 > q_1 and p_2 < q_2:
            return 1
        elif p_1 >= q_1 and p_2 < q_2:
            return 1
        elif p_1 > q_1 and p_2 <= q_2:
            return 1
        elif p_1 < q_1 and p_2 > q_2:
            return -1
        else:
            return 0

    def selectBynasg2(self):  # 非支配排序
        iter = 0
        S = [[] for i in range(len(self.forselect_pop))]
        front = [[]]
        n = [0 for i in range(len(self.forselect_pop))]  # n长度为len(population)，每个元素为0
        rank = [0 for i in range(len(self.forselect_pop))]
        # 求第0级前沿
        for p in range(len(self.forselect_pop)):  # 这里p q是int型
            for q in range(p + 1, len(self.forselect_pop)):
                flag = self.dominate(self.forselect_pop[p], self.forselect_pop[q])  # 这里可能耗时比较长
                if flag == 1:
                    S[p].append(q)
                    n[q] += 1
                if flag == -1:
                    S[q].append(p)
                    n[p] += 1
            if n[p] == 0:
                rank[p] = 0
                if p not in front[0]:
                    front[0].append(p)
                    self.gen_best_record[iter].append([temp_t1, temp_t2])  #
                    idx = front[0].index(p)

                    self.gen_solset[iter].append(self.forselect_pop[front[0][idx]])  # 每次都把第一级前沿的结果向量放入gen_solset

        i = 0
        while (front[i] != []):
            frontQ = []
            for p in front[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        frontQ.append(q)
            i += 1
            front.append(frontQ)
        Data_type = object  # 为适应python 3.8加的
        del [front[len(front) - 1]]
        return np.array(front, dtype=Data_type)

    def count_congestion(self, listforselect):  # list是第i级前沿的序号
        # 先对目标函数值排序
        clist = []
        cgest = np.zeros(shape=(len(listforselect)))
        for i in range(len(listforselect)):
            self.distance_counter.w_matrix = self.forselect_pop[i].copy()
            self.risk_counter.w_matrix = self.forselect_pop[i].copy()
            c1, _ = self.distance_counter.final_count()  # max
            c2 = self.risk_counter.final_count()  # min
            c = c1 + 10000 - c2  # 化为最大值
            clist.append(c)
        list_c = zip(listforselect, clist)  # 序号和函数值
        list_c = sorted(list_c, key=lambda s: -s[1])  # 默认为升序, 按照函数值排序; 改为降序
        list_c = list(list_c)
        # print("这是list_c",list_c)
        list_lidx = [j[0] for j in list_c]
        # 拥挤度
        # cgest[list_c[0][0]]=np.inf
        cgest[0] = np.inf
        # cgest[list_c[len(list)-1][0]] = np.inf
        cgest[len(listforselect) - 1] = np.inf
        for i in range(1, len(listforselect) - 1):
            cgest[i] = (list_c[i + 1][1] - list_c[i - 1][1]) / (list_c[len(listforselect) - 1][1] - list_c[0][1])

        cgest_l = zip(cgest, list_lidx)
        cgest_l = sorted(cgest_l, key=lambda s: s[0], reverse=True)
        cgest_lidx = [j[1] for j in cgest_l]
        return cgest_lidx

    def wake_s(self, w_list, idx):
        candidate = list(range(0, self.popsize))
        candidate.remove(idx)
        ram_idx = sample(candidate, 3)
        for i in range(0, 3):
            self.population[ram_idx[i]] = w_list

    def select_newpop(self, front):
        newpop = []  # 存储新一代种群个体的序号
        # while(len(newpop)<100):
        for i in range(len(front)):
            if (len(newpop + front[i]) < self.popsize):
                newpop += front[i]
            else:
                extra = self.popsize - len(newpop)
                cgl = self.count_congestion(front[i])
                newpop += cgl[0:extra]
        sums = 0
        sumr = 0

        for j in range(len(newpop)):
            self.distance_counter.w_matrix = self.forselect_pop[newpop[j]].copy()
            self.risk_counter.w_matrix = self.forselect_pop[newpop[j]].copy()

            a, _ = self.distance_counter.final_count()
            sums += a
            sumr += self.risk_counter.final_count()
            # print(sums, sumr)
            self.gens_scores.append(a)
            self.population[j] = self.forselect_pop[newpop[j]]

        self.gens_avg_record.append(sums / self.popsize)
        self.genr_avg_record.append(sumr / self.popsize)

    def evolution(self):
        iter = 0
        self.pop_init()
        for c in range(1, self.maxtier + 1):  # 每一轮都要更新population
            print("GENERATION:", c)
            self.gens_scores = []
            son_population = []
            for ind in range(self.popsize):
                candidates = list(range(0, self.popsize))
                candidates.remove(ind)  # 除去ind外，再任选3个个体
                random_index = sample(candidates, 3)
                # 先变异
                w1 = self.population[random_index[0]]
                w2 = self.population[random_index[1]]
                w3 = self.population[random_index[2]]
                w_t = self.population[ind]
                w_sub = [w2i - w3i for w2i, w3i in zip(w2, w3)]
                w_new = [w1i + self.mutate * w_subi for w1i, w_subi in zip(w1, w_sub)]
                # 对越界的进行处理
                w_new1 = []
                for num in range(len(w_new)):
                    if w_new[num] < self.bounds[0]:
                        w_new[num] = 0.01
                        w_new1.append(w_new[num])

                    if w_new[num] > self.bounds[1]:
                        w_new[num] = 5
                        w_new1.append(w_new[num])

                    if self.bounds[0] <= w_new[num] <= self.bounds[1]:
                        w_new1.append(w_new[num])

                # 交叉
                w_trial = []  # 就是一个子代
                for k in range(len(w_t)):
                    crossp = random()

                    if crossp < self.cr:  # 交叉
                        w_trial.append(w_new1[k])

                    else:
                        w_trial.append(w_t[k])
                if w_trial not in son_population:
                    son_population.append(w_trial)
            # 开始 NSGA2
            # 每个父本都产生了子代，之后父子代混合，进行选择
            self.forselect_pop = son_population + self.population
            self.addto1(self.forselect_pop)
            # print(forselect_pop)
            # print("Above is the pop")
            front = self.selectBynasg2()
            # print(front)
            self.select_newpop(front)
            # population = addto1(population)
            # print(gen_best_record[iter])
            if iter > 10:
                idx = self.gens_scores.index(max(self.gens_scores))
                self.wake_s(self.population[self.gens_scores.index(max(self.gens_scores))], idx)

            iter += 1

    def store_result(self):
        data = {
            "gen_best_record": self.gen_best_record,
            "gens_avg_record": self.gens_avg_record,
            "genr_avg_record": self.genr_avg_record
            # "gen_solset": self.gen_solset
        }
        # 写入到文件
        with open("./data_manipulate/double_opt.json", "w") as file:
            json.dump(data, file)
