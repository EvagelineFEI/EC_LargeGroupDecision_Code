# 计算similarity和 Sgroup
import math
from dataprocess import load_mat
import numpy as np
import time
from numba import cuda
from numba import jit
n=4
L=3
t=4
num=33 # 决策矩阵数目
alldata=load_mat()
w_matrix1=np.zeros(num) # 存放每个专家的意见权重

def initialw_m():
    pass

def insideProc(a):
    return (np.abs(a)/(2*t))**2
def theMultiadd(m1, m2):
    matrix_sub=m1-m2  # 2个矩阵对应位置上的元素相减
    proc=insideProc(matrix_sub)
    addAll=np.sum(proc)
    return addAll
def countSimilar(addall):
    return 1-math.sqrt(1/(L*n*n)*addall)

# @jit(nopython=True)
# @jit
def finalcount1():
  # time_start = time.time()
  storeDis1=np.zeros((num,num))
  for o1 in range(num-1): # 0~32
      matrix_1=alldata[o1]
      for o2 in range(o1+1,num): # 边界上可能有bug
          matrix_2=alldata[o2]
          addall=theMultiadd(matrix_1,matrix_2)
          siml=countSimilar(addall)
          storeDis1[o1][o2]=siml
  bf=np.zeros((4,4,3))
  for o1 in range(num):
      matrix=alldata[o1]
      bf += matrix*w_matrix1[o1]

  Sgroup = 0
  for o1 in range(num):
      matrix_g=alldata[o1]
      addall = theMultiadd(matrix_g, bf)
      siml1=countSimilar(addall)
      Sgroup+=siml1
  # time_end = time.time()
  # time_c = time_end - time_start
  # print("Time cost in calculateDistance is:",time_c)
  return Sgroup
# res=theMultiadd(alldata[0],alldata[1])
#print(res)
