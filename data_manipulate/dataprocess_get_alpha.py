import numpy as np
import os

current_path = os.getcwd()


def get_alpha(alpha_file):
    path = current_path + '\data_manipulate\\' + alpha_file
    alpha_ls = []
    with open(path,'r') as file:
        for line in file.readlines():
            value = line.strip().split()[0]
            alpha_ls.append(float(value))

    return np.array(alpha_ls)
