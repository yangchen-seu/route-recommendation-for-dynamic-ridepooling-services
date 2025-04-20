import pickle
import numpy as np
import time
import multiprocessing
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import Predict_class


# **✅ 全局变量**
pre = None
OD_dict = None

def objective_function(individual):
    theta_values = individual  # **直接使用 0/1**
    theta_dict = {OD_id: {'shortest_path': theta_values[i], 'highest_potential_path': 1 - theta_values[i]} 
                  for i, OD_id in enumerate(OD_dict.keys())}

    platform_profit, _ = pre.run(theta_dict)
    return (-platform_profit, )  # 负利润（最小化问题）
	
	
def run_ga():
    global pre, OD_dict  # **✅ 使用全局变量**
    # **读取数据**
    pre = Predict_class.Predict()
    with open("tmp/OD.pickle", 'rb') as f:
        OD_dict = pickle.load(f)

    num_ODs = len(OD_dict)
    num_dimensions = num_ODs  # 只优化 theta 参数


    # 初始化遗传算法
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", np.random.randint, 0, 11)  # 生成 0~10 之间的整数
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=num_dimensions)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", objective_function)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=10, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # ✅ **手动管理进程池**
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())  
    toolbox.register("map", pool.map)  # 替换默认 map

    # 运行遗传算法
    pop = toolbox.population(n=100)
    history = []

    begin = time.time()

    for gen in range(50):  # 运行 50 代
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)
        fits = toolbox.map(toolbox.evaluate, offspring)  # 并行计算

        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit

        pop = toolbox.select(offspring, k=len(pop))

        best = tools.selBest(pop, k=1)[0]
        history.append(best.fitness.values[0])  # 记录优化过程

        print(f"Generation {gen+1}, Best Function Value: {best.fitness.values[0]}")

    # ✅ **关闭进程池，防止 Python 退出时报错**
    pool.close()
    pool.join()

    # **最优解**
    best_solution = tools.selBest(pop, k=1)[0]
    best_theta = [x / 10 for x in best_solution]  # 归一化
    end = time.time()

    print("Best parameters:", best_theta)
    print('Executed time:', end - begin)

    # **画图**
    plt.plot(history)
    plt.xlabel('Generation')
    plt.ylabel('Objective Function Value')
    plt.title('Genetic Algorithm Optimization Progress')
    plt.savefig('result/Genetic Algorithm Optimization Progress.png')

if __name__ == '__main__':
    run_ga()
