# 双目标优化时运行此文件
import calculateDistance
import clculateRgroup
from calculateDistance import *
from clculateRgroup import *
from random import random
from random import sample
from random import uniform
import matplotlib.pyplot as plt
bounds=(0,1)  # 认为每个专家的意见权重都是大于0小于1的
popsize = 100
mutate = 0.5  # 变异策略中的缩放因子F ∈ [0,2]
recombination = 0.7  # 交叉概率CR ∈ [0,1]
maxiter = 300  # 最大迭代次数  0差别
iter=0
population=[]
# son_population=[]
forselect_pop=[]
gen_best_record=[[] for i in range(maxiter)]   #可视化: 每次迭代后，都显示出S和R的值
gen_solset = [[] for i in range(maxiter)]
w_tar=[0.4, 0.6]
temp_t1 = 0
temp_t2 = 0
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
        for q in range(len(forselect_pop)):
            if p != q:
                flag=dominate(forselect_pop[p],forselect_pop[q])
                if flag == 1:
                    S[p].append(q)
                if flag == -1:
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
# 种群初始化(每个个体是w_matrix)
for i in range(popsize):
   indiv=[]
   for j in range(num):
       indiv0=uniform(0,1)
       if indiv0 != 0:
         indiv.append(indiv0)
   population.append(indiv)

population = addto1(population)

for c in range(1,maxiter+1): # 每一轮都要更新population
    print("GENERATION:", c)
    gen_scores = []
    son_population = []
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
            if w_new[num]<bounds[0]:
                w_new[num] =0.01
                w_new1.append(w_new[num])

            if w_new[num]>bounds[1]:
                w_new[num]=5
                w_new1.append(w_new[num])

            if bounds[0]<= w_new[num]<= bounds[1]:
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

    # 每个父本都产生了子代，之后父子代混合，进行选择
    forselect_pop = son_population+population
    forselect_pop = addto1(forselect_pop)
    # print(forselect_pop)
    # print("Above is the pop")
    front = selectBynasg2(forselect_pop)
    # print(front)
    selectnewpop(front)
    print(gen_best_record[iter])
    iter += 1

        # file_obj.write('\n')
x_avr = []
y_avr = []
fig, axes = plt.subplots(1, 2)
for i in range(len(gen_best_record)):
    x = [j[0] for j in gen_best_record[i]]
    x_avr.append(sum(x)/len(x))
    # x = gen_best_record[i][0]
    y = [j[1] for j in gen_best_record[i]]
    print(y)
    y_avr.append(sum(y) / len(y))
    axes[0].scatter(x,y, label=f'data{i}')

axes[0].legend()
label = [i for i in range(1,18)]
for i in range(len(x_avr)):
    axes[1].scatter(x_avr,y_avr, c=label, cmap='viridis',alpha = 0.2)

for i in range(len(x_avr)):
    axes[1].text(x_avr[i],y_avr[i],i)
axes[1].legend()
plt.show()