import numpy as np


class Particle:
    def __init__(self, x_max, max_vel, dim, fit_func):
        self.__pos = np.random.uniform(-x_max, x_max, (1, dim))  # position of the particles
        self.__vel = np.random.uniform(-max_vel, max_vel, (1, dim))  # velocity of particles
        self.__bestPos = np.zeros((1, dim))
        self.__fitnessValue = fit_func(self.__pos)

    def set_pos(self, value):
        self.__pos = value

    def get_pos(self):
        return self.__pos

    def set_best_pos(self, value):
        self.__bestPos = value

    def get_best_pos(self):
        return self.__bestPos

    def set_vel(self, value):
        self.__vel = value

    def get_vel(self):
        return self.__vel

    def set_fitness_value(self, value):
        self.__fitnessValue = value

    def get_fitness_value(self):
        return self.__fitnessValue


class PSO:
    def __init__(self, dim, size, iter_num, x_max, max_vel, tol, fit_func, best_fitness_value=float('Inf'), C1=2, C2=2,
                 W=1):
        self.C1 = C1
        self.C2 = C2
        self.W = W
        self.dim = dim
        self.size = size
        self.iter_num = iter_num
        self.x_max = x_max
        self.max_vel = max_vel
        self.tol = tol
        self.fit_func = fit_func
        self.best_fitness_value = best_fitness_value
        self.best_position = np.zeros((1, dim))
        self.fitness_val_list = []

        self.Particle_list = [Particle(self.x_max, self.max_vel, self.dim, fit_func=self.fit_func) for i in
                              range(self.size)]

    def set_bestFitnessValue(self, value):
        self.best_fitness_value = value

    def get_bestFitnessValue(self):
        return self.best_fitness_value

    def set_bestPosition(self, value):
        self.best_position = value

    def get_bestPosition(self):
        return self.best_position

    def update_vel(self, part):
        vel_value = self.W * part.get_vel() + self.C1 * np.random.rand() * (part.get_best_pos() - part.get_pos()) \
                    + self.C2 * np.random.rand() * (self.get_bestPosition() - part.get_pos())
        vel_value[vel_value > self.max_vel] = self.max_vel
        vel_value[vel_value < -self.max_vel] = -self.max_vel
        part.set_vel(vel_value)

    def update_pos(self, part):
        pos_value = part.get_pos() + part.get_vel()
        part.set_pos(pos_value)
        value = self.fit_func(part.get_pos())
        if value < part.get_fitness_value():
            part.set_fitness_value(value)
            part.set_best_pos(pos_value)
        if value < self.get_bestFitnessValue():
            self.set_bestFitnessValue(value)
            self.set_bestPosition(pos_value)

    def update_ndim(self):

        for i in range(self.iter_num):
            for part in self.Particle_list:
                self.update_vel(part)
                self.update_pos(part)
            self.fitness_val_list.append(self.get_bestFitnessValue())
            print('iter: {}, best_value: {}'.format(i, self.get_bestFitnessValue()))
            if self.get_bestFitnessValue() < self.tol:
                break

        return self.fitness_val_list, self.get_bestPosition()
