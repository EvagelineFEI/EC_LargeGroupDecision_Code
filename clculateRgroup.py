# 计算风险值
import numpy as np
from dataprocess import load_mat
from numba import cuda
from numba import jit
n=4
L=3
t=4
num=33 # 决策矩阵数目
w_matrix2=np.zeros(num) # 存放每个专家的意见权重
def initialw_m():
   pass

# @jit(nopython=True)
# @jit
def countVar2t(t):
   sum=0
   sub1=2*t/2
   for sub0 in range(2*t+1):
     sum+=(sub0-sub1)**2
   return sum
# 操作对象是一个操作矩阵
# 先算出I(bijg) 存在一个矩阵里面
# Vijg应该存在一个4*4矩阵

# @jit(nopython=True)
# @jit
def countVgij(matrix_g):
    # m_gbij=np.zeros(4,4,1)
    proc=[]
    for i in range(t):
      m_gbij = np.sum(matrix_g[i], axis=1)
      proc.append(m_gbij)
    proc=np.array(proc)
    proc/=L

    matrix_vgij=np.zeros((n,n))
    for i in range(n):
       for j in range(n):
           inlinesum=0
           for l in range(L):
              inlinesum+=(matrix_g[i][j][l]-proc[i][j])**2
              Var2t=countVar2t(t)

           inlinesum=inlinesum/(L*Var2t)
           matrix_vgij[i][j]=proc[i][j]-inlinesum

    return matrix_vgij

def countVig(matrix_vgij):
    for i in range(n):
       matrix_vgij[i][i]=0
    matrix_vig=np.sum(matrix_vgij,axis=1)
    matrix_vig=np.array(matrix_vig)
    return matrix_vig

def countP(matrix_vig):
    belowsum=np.sum(matrix_vig)
    p=[]
    for i in range(n):
        p.append(matrix_vig[i]/belowsum)
    p=np.array(p)
    return p

def countE(mv,mp):   # mv即matrix_vig
    return np.multiply(mv,mp)

def countVar(e,matrix_vig):  # 这是计算单个值（每个决策矩阵的方差），最后要放入数组
    inlinesum=0
    for i in range(n):
        inlinesum+=(matrix_vig[i]-e[i])**2
    return inlinesum/n

def countCov(m_vig,m_vik,eg,ek): # 这是计算单个值（两个决策矩阵之间），最后要放入num*num矩阵
    m_vig-=eg
    m_vik-=ek
    mul=np.multiply(m_vig,m_vik)
    cov=0
    for i in range(n):
         cov+=mul[i]/n
    return cov
# @jit
def countR(mw,mvar,mcov):
    add1=0
    m_wvar=np.multiply(w_matrix2,mvar)
    for g in range(num):
        add1+=m_wvar[g]**2

    add2=0
    for g in range(num):
        for k in range(num):
            if k!=g:
                add2+=mw[g]*mw[k]*mcov[g][k]*mvar[g]*mvar[k]

    return add1+add2
# @jit
def finalcount2():
    alldata=load_mat()
    matrix_vgij=[]
    for i in range(num):
        matrix_vgij.append(countVgij(alldata[i]))

    allp=[]
    for i in range(num):
        m0=matrix_vgij[i]
        m_vig=countVig(m0)
        allp.append(countP(m_vig))
    alle=[]
    for i in range(num):
        m0 = matrix_vgij[i]
        m_vig = countVig(m0)
        p0=allp[i]
        alle.append(countE(m_vig,p0))

    allvar=[]
    for i in range(num):
        m_vig = countVig(matrix_vgij[i])
        allvar.append(countVar(alle[i],m_vig))
    mcov=np.zeros((num,num))
    for i in range(num):
        for j in range(num):
            m_vig = countVig(matrix_vgij[i])
            m_vik = countVig(matrix_vgij[j])
            mcov[i][j]=countCov(m_vig,m_vik,alle[i],alle[j])

    Rgroup=countR(w_matrix2,allvar,mcov)
    return Rgroup
# new_d=load_mat()
# res=countVgij(new_d[0])
# res1=countVig(res)
# res2=countP(res1)
# print(res1)
# print(res2)
