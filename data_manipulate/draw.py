import json

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter


def draw_single(testsingle):
    with open("./data_manipulate/single_opt.json", "r") as f:
        res = f.read()
        print("res-------------", type(res))
    data = json.loads(res)
    print("data-------------",type(data))
    gen_best_record = data['gen_best_record']
    gen_avg_record = data['gen_avg_record']
    maxtier = len(gen_avg_record)

    if testsingle == 1:
        figure_title1 = "The best S of \nevery generation"
        figure_title2 = "The average S of \nevery generation"
    else:
        figure_title1 = "The best R of \nevery generation"
        figure_title2 = "The average R of \nevery generation"
    iter = [i for i in range(1, maxtier + 1)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # 设置纵坐标格式
    # axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    for i in range(len(gen_best_record)):
        axes[0].scatter(iter[i], gen_best_record[i], color='orange', edgecolors='black')
    axes[0].tick_params(axis='both', which='major', labelsize=16)
    axes[0].set_xlabel("Generation", fontsize=20)
    # axes[0].set_ylabel("best", fontsize=16)
    axes[0].set_title(figure_title1, fontsize=20)
    axes[0].grid(True)
    # 设置纵坐标格式
    # axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    for i in range(len(gen_avg_record)):
        axes[1].scatter(iter[i], gen_avg_record[i], color='orange', edgecolors='black')
    axes[1].tick_params(axis='both', which='major', labelsize=16)
    axes[1].set_xlabel("Generation", fontsize=20)
    # axes[1].set_ylabel("average", fontsize=16)
    axes[1].set_title(figure_title2, fontsize=20)
    axes[1].grid(True)

    plt.tight_layout()
    if testsingle == 1:
        plt.savefig('D:\Desktop\论文\新版图片\s_single_newdata.png')
    else:
        plt.savefig('D:\Desktop\论文\新版图片\\r_single.png')

    plt.show()
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(False)
    for ax in axes:
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
