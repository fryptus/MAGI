import math
from random import random
import matplotlib.pyplot as plt


class SA:
    def __init__(self, obj_func, gen_solutions, init_sol, init_t=100, final_t=0.01, T0=100, alpha_t=0.99, step_l=100):
        self.obj_func = obj_func
        self.gen_solutions = gen_solutions
        self.init_sol = init_sol
        self.alpha_t = alpha_t
        self.T0 = init_t
        self.Tf = final_t
        self.T = T0
        self.step_l = step_l
        self.most_best = []
        self.history = {'best_f': [], 'T_now': []}

    def metrospolis_rule(self, f, f_new):  # Metropolis Rule
        if f_new <= f:
            return 1
        else:
            p = math.exp((f - f_new) / self.T)
            if random() < p:
                return 1
            else:
                return 0

    def best(self):  # get the best value of object function
        f_list = []
        for i in range(self.step_l):
            f = self.obj_func(self.init_sol[:, i])
            f_list.append(f)
        f_best = min(f_list)
        idx = f_list.index(f_best)
        return f_best, idx

    def run(self):  # run the sa
        count = 0
        while self.T > self.Tf:
            for i in range(self.step_l):
                f = self.obj_func(self.init_sol[:, i])
                sol_new = self.gen_solutions(self.T, self.init_sol[:, i])
                f_new = self.obj_func(sol_new)
                if self.metrospolis_rule(f, f_new):
                    self.init_sol[:, i] = sol_new
            ft, _ = self.best()
            self.history['best_f'].append(ft)
            self.history['T_now'].append(self.T)
            self.T = self.T * self.alpha_t
            count += 1
        f_best, idx = self.best()
        print(f"F={f_best}, solution={self.init_sol[:, idx]}")
        return f_best, idx

    def show_result_plot(self):
        plt.plot(self.history['T_now'], self.history['best_f'])
        plt.title('SA')
        plt.xlabel('T')
        plt.ylabel('f')
        plt.gca().invert_xaxis()
        plt.show()
