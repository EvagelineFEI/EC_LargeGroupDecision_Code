from optimize.doubleOptimize import DoubleOptimizer
from data_manipulate.dataprocess import load_mat
from data_manipulate.dataprocess_get_alpha import get_alpha
from data_manipulate.draw import draw_single

if __name__ == '__main__':
    num = 24
    data = load_mat()
    # data_new = load_mat_general(10,3,"example_M_repaired.txt")
    # print(data)
    alpha = get_alpha("Zong_alpha_list.txt")
    # print(len(alpha))
    # dimension = [10,10,3]
    optimizer = DoubleOptimizer(data=data, alpha=alpha)
    optimizer.evolution()
    optimizer.store_result()
    # draw_single(test_target)
