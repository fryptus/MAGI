import math
import numpy as np
from random import random
from magi_package.magi.simulated_annealing import SA


def func(init_solution):
    x = init_solution[0]
    y = init_solution[1]
    res = 4 * x ** 2 - 2.1 * x ** 4 + x ** 6 / 3 + x * y - 4 * y ** 2 + 4 * y ** 4
    return res


def generate_new(T, init_solution):
    x = init_solution[0]
    y = init_solution[1]
    while True:
        x_new = x + T * (random() - random())
        y_new = y + T * (random() - random())
        if (-5 <= x_new <= 5) & (-5 <= y_new <= 5):
            break  # 重复得到新解，直到产生的新解满足约束条件
    init_solution[0] = x_new
    init_solution[1] = y_new
    return init_solution


def init_solution(v_num, step_l):
    init_solution = np.zeros((v_num, step_l))
    for i in range(step_l):
        init_solution[0][i] = random() * 11 - 5
        init_solution[1][i] = random() * 11 - 5
    # print(init_solution)
    return init_solution


init_sol = init_solution(v_num=2, step_l=100)
print(init_sol.shape[0])
print(init_sol[:, 1])
print(init_sol[0])
sa = SA(func, gen_solutions=generate_new, init_sol=init_sol, step_l=100)
sa.run()
sa.show_result_plot()
