from random import random
from random import sample
from random import uniform
import numpy as np
from S_R_compute.distance import Distance
from S_R_compute.risk import GroupRisk
import json


class SingleOptimizer:
    def __init__(self, test_target, num, data, popsize=100, mutate=0.8, recombination=0.5, maxtier=100, elite=0.8):
        self.target = test_target
        self.popsize = popsize
        self.mutate = mutate
        self.cr = recombination
        self.maxtier = maxtier
        self.elite = elite
        self.population = []
        self.forselect_pop = []
        self.gen_avg_record = []
        self.gen_best_record = []  # 可视化: 每次迭代后，都显示出S和R的值
        self.gen_solset = []
        self.num = num
        self.bounds = (0, 1)
        self.w_matrix = np.zeros(num)
        self.distance_counter = Distance(data, self.w_matrix)
        self.risk_counter = GroupRisk(data, self.w_matrix)

    def addto1(self):
        for ind in range(len(self.population)):
            allsum = np.sum(self.population[ind])
            self.population[ind] /= allsum
        # return pop

    def pop_init(self):
        for i in range(self.popsize):
            indiv = []
            for j in range(self.num):
                indiv0 = uniform(0, 1)
                if indiv0 != 0 or indiv0 != 1:
                    indiv.append(indiv0)
            self.population.append(indiv)
            self.addto1()

    def wakeup(self, w_list, idx):
        candidates = list(range(0, self.popsize))
        candidates.remove(idx)
        random_index = sample(candidates, 3)
        l = len(w_list)
        for i in range(0, 3):
            # p = random() / 100
            # w_list[random_index[0]%l]+=p
            self.population[random_index[i]] = w_list

    def evolution(self):
        self.pop_init()
        cnt_max = 0
        best1 = 0
        for c in range(1, self.maxtier + 1):  # 每一轮都要更新population
            gen_scores = []
            son_population = []
            # if c == maxiter / 4:
            #     mutate += 0.2
            #     recombination += 0.3
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
                    if w_new[num] <= self.bounds[0]:
                        add = uniform(0, 1) / 5
                        w_new[num] = 0.001 + add
                        w_new1.append(w_new[num])

                    if w_new[num] >= self.bounds[1]:
                        add = uniform(0, 1) / 10
                        w_new[num] = 0.9 + add
                        w_new1.append(w_new[num])

                    if self.bounds[0] < w_new[num] < self.bounds[1]:
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

                if self.target == 1:  # 只考虑意见相似度
                    self.distance_counter.w_matrix = w_trial.copy()
                    score_trial, _ = self.distance_counter.final_count()
                    self.distance_counter.w_matrix = w_t.copy()
                    score_t, _ = self.distance_counter.final_count()

                    if score_trial > score_t:
                        # if p <= elite:  # 保留精英
                        self.population[ind] = w_trial.copy()
                        gen_scores.append(score_trial)
                    # else:
                    #     gen_scores.append(score_t)
                    else:
                        # if p <= elite: # 保留精英
                        gen_scores.append(score_t)
                    # else:
                    #     population[ind] = w_trial
                    #     gen_scores.append(score_trial)
                elif self.target == 2:  # 只考虑风险值
                    self.risk_counter.w_matrix = w_trial.copy()
                    score_trial = self.risk_counter.final_count()

                    self.risk_counter.w_matrix = w_t.copy()
                    score_t = self.risk_counter.final_count()
                    # p = random()
                    if score_trial < score_t:
                        # if p <= elite:  # 保留精英
                        self.population[ind] = w_trial.copy()
                        gen_scores.append(score_trial)
                    # else:
                    #     gen_scores.append(score_t)
                    else:
                        # if p <= elite:  # 保留精英
                        gen_scores.append(score_t)
                    # else:
                    #     population[ind] = w_trial.copy()
                    #     gen_scores.append(score_trial)
            gen_avg = sum(gen_scores) / self.popsize
            self.gen_avg_record.append(gen_avg)

            if self.target == 1:
                gen_best = max(gen_scores)  # fitness of best individual
                if gen_best == best1:
                    cnt_max += 1
                else:
                    best1 = gen_best
                self.gen_best_record.append(gen_best)
                self.gen_solset.append(self.population[gen_scores.index(max(gen_scores))])
                # print(gen_scores.index(max(gen_scores)))
                if cnt_max > 10:
                    idx = gen_scores.index(max(gen_scores))
                    self.wakeup(self.population[gen_scores.index(max(gen_scores))], idx)

            elif self.target == 2:
                gen_best = min(gen_scores)  # fitness of best individual
                self.gen_best_record.append(gen_best)
                self.gen_solset.append(self.population[gen_scores.index(min(gen_scores))])
                gen_sol = self.population[gen_scores.index(min(gen_scores))]

    def store_result(self):

        data = {
            "gen_best_record": self.gen_best_record,
            "gen_avg_record": self.gen_avg_record
            # "gen_solset": self.gen_solset
        }
        # 写入到文件
        with open("./data_manipulate/single_opt.json", "w") as file:
            json.dump(data, file)
