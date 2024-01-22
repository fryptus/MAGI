from math import floor
import numpy as np
import time
import matplotlib.pyplot as plt


class GA(object):

    def __init__(self, data, maxgen=1000, size_pop=100, cross_prob=0.80, pmuta_prob=0.02, select_prob=0.8):
        self.maxgen = maxgen  # maximum number of iterations
        self.size_pop = size_pop  # size of population
        self.cross_prob = cross_prob  # cross prob
        self.pmuta_prob = pmuta_prob  # mutate prob
        self.select_prob = select_prob  # select prob

        self.data = data
        self.num = len(data)  # chromosome length
        # the [i,j] element represents the distance from city i to j. The matrix_dis function is described below.
        self.matrix_distance = self.matrix_dis()

        # determine the number of the children
        self.select_num = max(floor(self.size_pop * self.select_prob + 0.5), 2)

        # init the population
        self.chrom = np.array([0] * self.size_pop * self.num).reshape(self.size_pop,
                                                                      self.num)  # 父 print(chrom.shape)(200, 14)
        self.sub_sel = np.array([0] * int(self.select_num) * self.num).reshape(self.select_num, self.num)  # 子 (160, 14)

        self.fitness = np.zeros(self.size_pop)

        self.best_fit = []
        self.best_path = []

    # calculate the distance between cities
    def matrix_dis(self):
        res = np.zeros((self.num, self.num))
        for i in range(self.num):
            for j in range(i + 1, self.num):
                res[i, j] = np.linalg.norm(self.data[i, :] - self.data[j, :])  # 求二阶范数 就是距离公式
                res[j, i] = res[i, j]
        return res

    # random generation of initialized population
    def rand_chrom(self):
        rand_ch = np.array(range(self.num))
        for i in range(self.size_pop):
            np.random.shuffle(rand_ch)
            self.chrom[i, :] = rand_ch
            self.fitness[i] = self.comp_fit(rand_ch)

    # calculate path distance values for individual chromosomes
    def comp_fit(self, one_path):
        res = 0
        for i in range(self.num - 1):
            res += self.matrix_distance[one_path[i], one_path[i + 1]]
        res += self.matrix_distance[one_path[-1], one_path[0]]
        return res

    def out_path(self, one_path):
        res = str(one_path[0] + 1) + '-->'
        for i in range(1, self.num):
            res += str(one_path[i] + 1) + '-->'
        res += str(one_path[0] + 1) + '\n'
        print(res)

    # subgeneration selection, using a random traversal selection method
    def select_sub(self):
        fit = 1. / (self.fitness)
        cumsum_fit = np.cumsum(fit)
        pick = cumsum_fit[-1] / self.select_num * (
                np.random.rand() + np.array(range(int(self.select_num))))
        i, j = 0, 0
        index = []
        while i < self.size_pop and j < self.select_num:
            if cumsum_fit[i] >= pick[j]:
                index.append(i)
                j += 1
            else:
                i += 1
        self.sub_sel = self.chrom[index, :]  # chrom 父

    # crossover
    def cross_sub(self):
        if self.select_num % 2 == 0:  # select_num160
            num = range(0, int(self.select_num), 2)
        else:
            num = range(0, int(self.select_num - 1), 2)
        for i in num:
            if self.cross_prob >= np.random.rand():
                self.sub_sel[i, :], self.sub_sel[i + 1, :] = self.intercross(self.sub_sel[i, :], self.sub_sel[i + 1, :])

    def intercross(self, ind_a, ind_b):
        r1 = np.random.randint(self.num)
        r2 = np.random.randint(self.num)
        while r2 == r1:  # 如果r1==r2
            r2 = np.random.randint(self.num)
        left, right = min(r1, r2), max(r1, r2)
        ind_a1 = ind_a.copy()
        ind_b1 = ind_b.copy()
        for i in range(left, right + 1):
            ind_a2 = ind_a.copy()
            ind_b2 = ind_b.copy()
            ind_a[i] = ind_b1[i]
            ind_b[i] = ind_a1[i]
            x = np.argwhere(ind_a == ind_a[i])
            y = np.argwhere(ind_b == ind_b[i])
            if len(x) == 2:
                ind_a[x[x != i]] = ind_a2[i]
            if len(y) == 2:
                ind_b[y[y != i]] = ind_b2[i]
        return ind_a, ind_b

    # mutate
    def mutation_sub(self):
        for i in range(int(self.select_num)):
            if np.random.rand() <= self.cross_prob:
                r1 = np.random.randint(self.num)
                r2 = np.random.randint(self.num)
                while r2 == r1:
                    r2 = np.random.randint(self.num)
                self.sub_sel[i, [r1, r2]] = self.sub_sel[i, [r2, r1]]

    # evolutionary reversal
    def reverse_sub(self):
        for i in range(int(self.select_num)):
            r1 = np.random.randint(self.num)
            r2 = np.random.randint(self.num)
            while r2 == r1:
                r2 = np.random.randint(self.num)
            left, right = min(r1, r2), max(r1, r2)
            sel = self.sub_sel[i, :].copy()

            sel[left:right + 1] = self.sub_sel[i, left:right + 1][::-1]
            if self.comp_fit(sel) < self.comp_fit(self.sub_sel[i, :]):
                self.sub_sel[i, :] = sel

    def reins(self):
        index = np.argsort(self.fitness)[::-1]
        self.chrom[index[:self.select_num], :] = self.sub_sel

    def run(self):
        self.rand_chrom()
        data = self.data
        fig, ax = plt.subplots()
        x = data[:, 0]
        y = data[:, 1]
        ax.scatter(x, y, linewidths=0.1)
        for i, txt in enumerate(range(1, len(data) + 1)):
            ax.annotate(txt, (x[i], y[i]))
        res0 = self.chrom[0]
        x0 = x[res0]
        y0 = y[res0]
        for i in range(len(data) - 1):
            plt.quiver(x0[i], y0[i], x0[i + 1] - x0[i], y0[i + 1] - y0[i], color='r', width=0.005, angles='xy', scale=1,
                       scale_units='xy')
        plt.quiver(x0[-1], y0[-1], x0[0] - x0[-1], y0[0] - y0[-1], color='r', width=0.005, angles='xy', scale=1,
                   scale_units='xy')
        plt.show()
        print('initial path: ' + str(self.fitness[0]))

        for i in range(self.maxgen):
            self.select_sub()
            self.cross_sub()
            self.mutation_sub()
            self.reverse_sub()
            self.reins()

            for j in range(self.size_pop):
                self.fitness[j] = self.comp_fit(self.chrom[j, :])

            index = self.fitness.argmin()
            if (i + 1) % 50 == 0:
                timestamp = time.time()
                formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
                print(formatted_time)
                print(str(i + 1) + " gen's" + ' best length: ' + str(self.fitness[index]))
                print(str(i + 1) + " gen's" + ' beat path:')
                self.out_path(self.chrom[index, :])

            self.best_fit.append(self.fitness[index])
            self.best_path.append(self.chrom[index, :])

        res1 = self.chrom[0]
        x0 = x[res1]
        y0 = y[res1]
        for i in range(len(data) - 1):
            plt.quiver(x0[i], y0[i], x0[i + 1] - x0[i], y0[i + 1] - y0[i], color='r', width=0.005, angles='xy', scale=1,
                       scale_units='xy')
        plt.quiver(x0[-1], y0[-1], x0[0] - x0[-1], y0[0] - y0[-1], color='r', width=0.005, angles='xy', scale=1,
                   scale_units='xy')
        plt.show()
