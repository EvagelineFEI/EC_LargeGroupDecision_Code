from optimize.singleOptimize import SingleOptimizer
from data_manipulate.dataprocess import load_mat,load_mat_general
from data_manipulate.dataprocess_get_alpha import get_alpha
from data_manipulate.draw import draw_single

if __name__ == '__main__':
    test_target = 1
    num = 24
    data = load_mat()
    data_new = load_mat_general(10,3,"example_M_repaired.txt")
    print(data)
    # alpha = get_alpha("example_alpha.txt")
    # print(len(alpha))
    # dimension = [10,10,3]
    # optimizer = SingleOptimizer(test_target, num, data_new,alpha, dimension,popsize=100)
    # optimizer.evolution()
    # optimizer.store_result()
    # draw_single(test_target)
