from method_for_comparison.com_second import IntervalTypeFuzzy
from data_manipulate.dataprocess import load_mat
from data_manipulate.dataprocess_get_alpha import get_alpha
if __name__ == '__main__':
    data = load_mat()
    # alpha = get_alpha("Zong_alpha_list.txt")
    inter_fuzzy = IntervalTypeFuzzy(data=data)
    res = inter_fuzzy.count_r()
    print(res)