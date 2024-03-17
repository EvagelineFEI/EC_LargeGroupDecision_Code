# process the data in file Zong_M_repaired_list.txt
import numpy as np
import os
current_path = os.getcwd()
cnt = 0


def load_mat():
    global cnt
    dataMat = []
    fr = open(current_path+'\data_manipulate'+'\Zong_M_repaired_list.txt')
    for line in fr.readlines():  # 第一个循环，把数据三个为一行，存到新矩阵里
        L1 = []  # 暂存4*4*3中的第一行（即第二层）
        curLine = line.strip().split()
        curLine = np.array(curLine)
        fltLine = list(map(float, curLine))  # map all elements to float() map:对第二个参数里面的每个元素使用第一个函数处理
        L1.append(fltLine[0:3])
        L1.append(fltLine[3:6])
        L1.append(fltLine[6:9])
        L1.append(fltLine[9:12])
        dataMat.append(L1)
        L2 = []  # 暂存4*4*3中的第二行（即第二层）
        L2.append(fltLine[12:15])
        L2.append(fltLine[15:18])
        L2.append(fltLine[18:21])
        L2.append(fltLine[21:24])
        dataMat.append(L2)
        L3 = []  # 暂存4*4*3中的第三行（即第二层）
        L3.append(fltLine[24:27])
        L3.append(fltLine[27:30])
        L3.append(fltLine[30:33])
        L3.append(fltLine[33:36])
        dataMat.append(L3)
        L4 = []  # 暂存4*4*3中的第四行（即第二层）
        L4.append(fltLine[36:39])
        L4.append(fltLine[39:42])
        L4.append(fltLine[42:45])
        L4.append(fltLine[45:48])
        dataMat.append(L4)  # L1,L2,L3,L4依序存进dataMat里
    mat = []  # 暂时存储一个决策矩阵(即第三层)
    new_data = []  # 存储一个txt里的所有决策矩阵

    for i in range(len(dataMat)):  # 每四行为一个决策矩阵，存到最后的矩阵
        if i < len(dataMat) - 1:  # 最后一个需要添加dataMat[1999]后立即添加到new_data里，与众不同，所以得单独提出来放在else里面
            if len(mat) < 4:
                mat.append(dataMat[i])
            else:
                new_data.append(mat)
                cnt = cnt + 1
                mat = []  # 存好一个决策矩阵至new_data后，放空，准备存储下一个到new_data
                mat.append(dataMat[i])
        else:
            mat.append(dataMat[i])
            new_data.append(mat)
            cnt = cnt + 1
    return np.array(new_data)


# new_d = load_mat()
# # proc=[]
# # for i in range(4):
# #     m_gbij = np.sum(new_d[0][i], axis=1)
# #     proc.append(m_gbij)
# # proc=np.array(proc)
# # proc/=3
# # print(proc)
# # m_gbij=np.sum(new_d[0],axis=0)
# # print(m_gbij)
# print(new_d)
# # print(cnt)
# # if __name__ == '__main__':
# #     new_d=load_mat()
# #     print(new_d)
