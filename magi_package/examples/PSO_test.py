import matplotlib.pyplot as plt
from magi_package.magi.particle_swarm_optimization import PSO


def fit_fun(x):  # fitness function
    return sum(100.0 * (x[0][1:] - x[0][:-1] ** 2.0) ** 2.0 + (1 - x[0][:-1]) ** 2.0)


pso = PSO(4, 5, 10000, 30, 60, 1e-4, fit_func=fit_fun, C1=2, C2=2, W=1)
fit_var_list, best_pos = pso.update_ndim()
print("position_best:" + str(best_pos))
print("result_best:" + str(fit_var_list[-1]))
plt.plot(range(len(fit_var_list)), fit_var_list, alpha=0.5)
plt.show()
