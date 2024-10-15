# Evolutionary Computation in the Scenario of Large-Group-Decision

## Intro项目简介

According to the theory of hesitation-fuzzy language with granularity, the formulas of expert consensus degree **S** and group decision risk degree **R** in a large group decision making problem are obtained.

We need to assign weights to each expert's opinion in order to make optimal decisions, and the optimal weights should make **S** as large as possible and **R** as small as possible.

This project uses differential evolution algorithm to optimize this problem; The problems are solved under the background of single objective and double objective respectively. The NSGA II algorithm is used to solve the double objective problem.

This project is part of the paper，[Biobjective Optimization Method for Large-Scale Group Decision Making Based on Hesitant Fuzzy Linguistic Preference Relations With Granularity Levels](https://ieeexplore.ieee.org/document/10557532)

根据含粒度的犹豫模糊语言理论，得出一个大群体决策问题中，各个专家意见共识度**S**和群体决策风险度**R**的计算公式。

我们需要为每个专家的意见分配权重，以便择优决策，最优的权重应该使得**S**尽可能大，**R**尽可能小。

本项目使用差分进化算法来优化这个问题；并分别在单目标和双目标背景下解决问题。解决双目标问题时使用了NSGAⅡ算法。

本项目是论文Biobjective Optimization Method for Large-Scale Group Decision Making Based on Hesitant Fuzzy Linguistic Preference Relations With Granularity Levels的一部分，用于解决论文中的双目标优化部分，帮助用户找到最佳的专家意见权重，使得群体决策的风险值和决策一致性都达到最优。

# 

## Framework代码框架介绍

Run run_double.py and run_single.py to optimize; And visualize the results. The results of the optimization are written to the json file under the data_manipulate folder.

The S_R_compute folder contains the classes that compute **S** and **R**; Under the optimize folder are the optimization classes.

Disadvantages: Double objective optimization, if the popsize is set to a larger number, such as 100, the running round is set to 100, the running time will be very long, and the relevant part of the code needs to be optimized.

运行run_double.py和run_single.py即可进行优化；并体现结果的可视化结果。优化的结果会写入data_manipulate文件夹下面的json文件。

S_R_compute文件夹下是计算**S**和**R**的类；optimize文件夹下是优化类。

缺点：双目标优化，如果popsize设置大一些，比如100，运行轮次设置为100，运行时间会很长，相关部分的代码有待优化。
