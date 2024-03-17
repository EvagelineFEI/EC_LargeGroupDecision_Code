from optimize.singleOptimize import SingleOptimizer
from data_manipulate.dataprocess import load_mat
from data_manipulate.draw import draw_single

if __name__ == '__main__':
    test_target = 1
    num = 30
    data = load_mat()
    optimizer = SingleOptimizer(test_target, num, data, popsize=100)
    optimizer.evolution()
    optimizer.store_result()
    draw_single(test_target)
