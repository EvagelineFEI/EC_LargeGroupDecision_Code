# 单目标优化时运行此文件
import numpy as np
import calculateDistance
import clculateRgroup
from calculateDistance import *
from clculateRgroup import *
from random import random
from random import sample
from random import uniform
import matplotlib.pyplot as plt
testsingle=1  # 1 优化S; 2 优化R
bounds=(0,1)  # 认为每个专家的意见权重都是大于0小于1的
popsize = 50
mutate = 0.5  # 变异策略中的缩放因子F ∈ [0,2]
recombination = 0.5  # 交叉概率CR ∈ [0,1]
maxiter = 100  # 最大迭代次数  0差别
elite = 0.8 # 精英保留概率
iter=0
population=[]
# son_population=[]
forselect_pop=[]
gen_avg_record=[]
gen_best_record=[]   # 可视化: 每次迭代后，都显示出S和R的值
gen_solset = []
w_tar=[0.4, 0.6]
temp_t1 = 0
temp_t2 = 0
cnt_max = 0
best1 = 0
# 归一化,使所有权重值相加是1
def addto1(pop):
    for ind in range(len(pop)):
        allsum = np.sum(pop[ind])
        pop[ind] /= allsum
    return pop

# @jit(nopython=True)
def dominate(p,q): # p q 是两个向量,计算 p 是否支配 q
    calculateDistance.w_matrix1 = clculateRgroup.w_matrix2 = p
    p_1 = finalcount1()
    p_2 = finalcount2()
    calculateDistance.w_matrix1 = clculateRgroup.w_matrix2 = q
    q_1 = finalcount1()
    q_2 = finalcount2()
    global temp_t1
    temp_t1 = p_1
    global temp_t2
    temp_t2 = p_2
    if p_1>q_1 and p_2<q_2:
        return 1
    elif p_1<q_1 and p_2>q_2:
        return -1
    else:
        return 0
# @jit(nopython=True)
def selectBynasg2(forselect_pop):   # 非支配排序
    S=[[] for i in range(len(forselect_pop))]
    front=[[]]
    n = [0 for i in range(len(forselect_pop))]  # n长度为len(population)，每个元素为0
    rank = [0 for i in range(len(forselect_pop))]
    # 求第0级前沿
    for p in range(len(forselect_pop)):  #这里p q是int型
        for q in range(p + 1, len(forselect_pop)):
            flag = dominate(forselect_pop[p], forselect_pop[q])  # 这里可能耗时比较长
            T10 = time.time()
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
               gen_best_record[iter].append([temp_t1, temp_t2]) #
               idx = front[0].index(p)
               gen_solset[iter].append(forselect_pop[front[0][idx]])  # 每次都把第一级前沿的结果向量放入gen_solset

    i=0
    while(front[i] != []):
        frontQ=[]
        for p in front[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i+1
                    frontQ.append(q)
        i += 1
        front.append(frontQ)

    del[front[len(front)-1]]
    return np.array(front)

def countCongestion(listforselect):  # list是第i级前沿的序号
    # 先对目标函数值排序
    clist = []
    cgest = np.zeros(shape=(len(listforselect)))
    for i in range(len(listforselect)):
        calculateDistance.w_matrix1 = clculateRgroup.w_matrix2 = forselect_pop[i]
        c1 = finalcount1()  # max
        c2 = finalcount2()  # min
        c = c1 + 100 - c2  # 化为最大值
        clist.append(c)
    list_c = zip(listforselect, clist) # 序号和函数值
    list_c = sorted(list_c, key=lambda s: s[1]) # 默认为升序, 按照函数值排序
    list_c = list(list_c)
    print("这是list_c",list_c)
    list_lidx = [j[0] for j in list_c]
    # 拥挤度
    # cgest[list_c[0][0]]=np.inf
    cgest[0] = np.inf
    # cgest[list_c[len(list)-1][0]] = np.inf
    cgest[len(listforselect) - 1] = np.inf
    for i in range(1, len(listforselect)-1):
         # cgest[list_c[i][0]] = (list_c[i+1][1] - list_c[i-1][1])/(list_c[len(list)-1][1]-cgest[list_c[0][1]])
         cgest[i] = (list_c[i + 1][1] - list_c[i - 1][1]) / (list_c[len(listforselect) - 1][1] - list_c[0][1])

    cgest_l=zip(cgest,list_lidx)
    cgest_l=sorted(cgest_l, key=lambda s: s[0], reverse = True)
    cgest_lidx=[j[1] for j in cgest_l]
    return cgest_lidx
def selectnewpop(front):
    newpop=[]  # 存储新一代种群个体的序号
    # while(len(newpop)<100):
    for i in range(len(front)):
        if(len(newpop+front[i])<popsize):
            newpop += front[i]
        else:
            extra = popsize-len(newpop)
            cgl=countCongestion(front[i])
            newpop += cgl[0:extra]

    for j in range(len(newpop)):
            population[j] = forselect_pop[newpop[j]]

def wakeup(w_list,idx):
    # l = len(w_list)
    # for i in range(l):
    #    if i/2==0:p
    #        p=random()/100
    #        q=random()
    #        if q<=0.6:
    #          w_list[i]+=p
    candidates = list(range(0, popsize))
    candidates.remove(idx)
    random_index = sample(candidates, 3)
    l = len(w_list)
    for i in range(0,3):
        # p = random() / 100
        # w_list[random_index[0]%l]+=p
        population[random_index[0]] = w_list
# 种群初始化(每个个体是w_matrix)
for i in range(popsize):
   indiv=[]
   for j in range(num):
       indiv0=uniform(0,1)
       if indiv0 != 0 or indiv0 != 1:
         indiv.append(indiv0)
   population.append(indiv)

population = addto1(population)
# print(population)

for c in range(1,maxiter+1): # 每一轮都要更新population
    # print("GENERATION:", c)
    gen_scores = []
    son_population = []
    if c == maxiter/4 :
        mutate += 0.2
        recombination += 0.3
    for ind in range(popsize):
        candidates = list(range(0, popsize))
        candidates.remove(ind)  # 除去ind外，再任选3个个体
        random_index = sample(candidates, 3)
        # 先变异
        w1 = population[random_index[0]]
        w2 = population[random_index[1]]
        w3 = population[random_index[2]]
        w_t = population[ind]

        w_sub = [w2i - w3i for w2i, w3i in zip(w2, w3)]
        w_new = [w1i + mutate * w_subi for w1i, w_subi in zip(w1, w_sub)]
        # 对越界的进行处理
        w_new1 = []
        for num in range(len(w_new)):
            if w_new[num]<=bounds[0]:
                add = uniform(0, 1)/5
                w_new[num] =0.001+add
                w_new1.append(w_new[num])

            if w_new[num]>=bounds[1]:
                add = uniform(0, 1) / 10
                w_new[num]=0.9+add
                w_new1.append(w_new[num])

            if bounds[0] < w_new[num] < bounds[1]:
                w_new1.append(w_new[num])

        # 交叉
        w_trial = [] #就是一个子代
        for k in range(len(w_t)):
            crossp = random()
            if crossp < recombination:  # 交叉
                w_trial.append(w_new1[k])
            else:
                w_trial.append(w_t[k])
        if w_trial not in son_population:
            son_population.append(w_trial)

        if testsingle==1:  # 只考虑意见相似度
            calculateDistance.w_matrix1=w_trial
            score_trial = finalcount1()
            calculateDistance.w_matrix1 = w_t
            score_t= finalcount1()

            if score_trial > score_t:
                # if p <= elite:  # 保留精英
                    population[ind] = w_trial
                    gen_scores.append(score_trial)
                # else:
                #     gen_scores.append(score_t)
            else:
                # if p <= elite: # 保留精英
                    gen_scores.append(score_t)
                # else:
                #     population[ind] = w_trial
                #     gen_scores.append(score_trial)
        if testsingle == 2:  # 只考虑意见相似度
            clculateRgroup.w_matrix2 = w_trial
            score_trial = finalcount2()

            clculateRgroup.w_matrix2 = w_t
            score_t = finalcount2()
            p = random()
            if score_trial < score_t:
                if p <= elite:  # 保留精英
                    population[ind] = w_trial
                    gen_scores.append(score_trial)
                else:
                    gen_scores.append(score_t)
            else:
                if p <= elite:  # 保留精英
                    gen_scores.append(score_t)
                else:
                    population[ind] = w_trial
                    gen_scores.append(score_trial)
    gen_avg = sum(gen_scores) / popsize
    gen_avg_record.append(gen_avg)
    if testsingle == 1:
        gen_best = max(gen_scores)  # fitness of best individual
        if gen_best == best1:
            cnt_max += 1
        else:
            best1=gen_best
        gen_best_record.append(gen_best)
        gen_solset.append(population[gen_scores.index(max(gen_scores))])
        # print(gen_scores.index(max(gen_scores)))
        if cnt_max > 10:
           idx = gen_scores.index(max(gen_scores))
           wakeup(population[gen_scores.index(max(gen_scores))],idx)

    if testsingle == 2:
        gen_best = min(gen_scores)  # fitness of best individual
        gen_best_record.append(gen_best)
        gen_solset.append(population[gen_scores.index(min(gen_scores))])
        gen_sol = population[gen_scores.index(min(gen_scores))]

print(gen_best_record)
print(gen_avg_record)
iter=[i for i in range(1,maxiter+1)]
fig, axes = plt.subplots(1,2)
for i in range(len(gen_best_record)):
    axes[0].scatter(iter[i],gen_best_record[i], label=f'data{i}')
axes[0].legend()

for i in range(len(gen_avg_record)):
    axes[1].scatter(iter[i],gen_avg_record[i], label=f'data{i}')

axes[1].legend()
plt.show()

print(mutate)
print(recombination)

# for i in range(len(gen_solset)):
#     print(gen_solset[i])

