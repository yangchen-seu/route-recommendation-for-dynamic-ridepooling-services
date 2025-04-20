import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import Predict_class
import json
import os
import multiprocessing



# **✅ 目标函数**
def objective_function(individual, OD_dict, pre, params):
    global best_result  # 使用全局变量

    theta_values = individual  # **直接使用 0/1**
    
    theta_dict = {OD_id: {'shortest_path': theta_values[OD_id], 'highest_potential_path': 1 - theta_values[OD_id]} 
                  for OD_id in OD_dict.keys()}

    try:
        platform_profit, result = pre.run(theta_dict, params)
    except Exception as e:
        print(f"Error in pre.run(): {e}")
        import traceback
        traceback.print_exc()  # 打印完整错误信息
        return (float('inf'), )  # 返回极大的值，避免影响优化

    return (-platform_profit, )  # 负利润（最小化问题）


def run_ga(OD_dict, params):
    # **读取数据**
    pre = Predict_class.Predict()
    num_ODs = len(OD_dict)
    num_dimensions = num_ODs  # 只优化 theta 参数

    # **初始化遗传算法**
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", np.random.randint, 0, 2)  # **0/1 变量**
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=num_dimensions)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", objective_function, OD_dict=OD_dict, pre=pre, params = params)  # 显式传递 OD_dict

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)  # **翻转 0/1**
    toolbox.register("select", tools.selTournament, tournsize=3)

    # **运行遗传算法**
    pop = toolbox.population(n=100)
    history = []

    begin = time.time()

    # **使用 multiprocessing 加速评估**
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())  # 进程池
    toolbox.register("map", pool.map)  # 让 DEAP 使用多进程 map

    
    for gen in range(50):  # 运行 50 代
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)
        fits = toolbox.map(toolbox.evaluate, offspring)  # **并行计算适应度**
        
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit

        pop = toolbox.select(offspring, k=len(pop))

        best = tools.selBest(pop, k=1)[0]
        history.append(best.fitness.values[0])  # **记录优化过程**
        print(f"Generation {gen+1}, Best Function Value: {best.fitness.values[0]}")

    pool.close()  # 关闭进程池
    pool.join()   # 等待所有进程结束

    # **最优解**
    best_solution = tools.selBest(pop, k=1)[0]
    best_theta = best_solution  # **不需要归一化**
    end = time.time()

    print("Best parameters:", best_theta)
    print('Executed time:', end - begin)

    # **画图**
    os.makedirs("result", exist_ok=True)
    plt.plot(history)
    plt.xlabel('Generation')
    plt.ylabel('Objective Function Value')
    plt.title('Genetic Algorithm Optimization Progress')
    plt.savefig('result/Genetic Algorithm Optimization Progress.png')

    # print('self.result',pre.result.head())

    return best_theta, pre.result  # 返回最优解


if __name__ == '__main__':
    setting_path = "settings.py"
    hours = [i for i in range(6, 24)]
    start = time.time()

    for hour in hours:
        begin = time.time()
        print(f"Processing hour: {hour}")
        # 构造 params 字典
        params = {
            'lowest_road_class': 5,
            'max_combined_length': 1000,
            'OD_num': 291,
            'prediction_results_file': f'./optimized/result/predicted_result_hour_{hour}.csv',
            'process_num': 32,
            'chunk_num': 5,
            'search_radius': 1000,
            'max_detour': 1000,
            'w_detour': 0,
            'w_pickup': 0,
            'w_shared': 0,
            "w_ride": -1,
            'pickup_time': 2,
            'speed': 600,  # m/min
            'max_iter_time': 100,
            "min_iter_time": 5,
            'M': 1e6,
            'epsilon': 1e-4,
            'n_v': 300,
            'beta': 1,
            'delta': 1,
            'K': 300,
            'OD_pickle_file': f'./result/different_hour/0_shortest_path/tmp/OD_hour_{hour}.pickle',
            'graph_pickle_file': './tmp/graph.pickle',
            'shortest_path_pickle': f"./optimized/tmp/shortest_path_hour_{hour}.pickle",
            'ego_graph_pickle': f"./optimized/tmp/ego_graph_hour_{hour}.pickle",
            'match_csv': f"./optimized/tmp/match_hour_{hour}.csv",
            'node_pickle': "./tmp/node.pickle",
            'link_pickle': "./tmp/link.pickle",
            'node_pickle_file':'./tmp/node.pickle',
        }

        # 生成格式化的 setting.py 内容
        new_setting_content = "params = " + json.dumps(params, indent=4)

        # 写回 setting.py
        with open(setting_path, "w", encoding="utf-8") as f:
            f.write(new_setting_content)

        with open(params['OD_pickle_file'], 'rb') as f:
            OD_dict: dict = pickle.load(f)

        # 运行遗传算法并获取最优解
        best_theta, result = run_ga(OD_dict, params)
        end = time.time()
        print(f"Best solution for hour {hour}: {best_theta}, time used {end - begin}")
        with open(f'./optimized/result/best_theta_hour_{hour}.pkl', 'wb') as f:
            pickle.dump(best_theta, f)

    # setting_path = "settings.py"
    # dates = [i for i in range(1, 22)]
    # start = time.time()

    # for date in dates:
    #     begin = time.time()
    #     print(f"Processing Date: {date}")
        
    #     # 构造 params 字典
    #     params = {
    #         'lowest_road_class': 5,
    #         'max_combined_length': 1000,
    #         'OD_num': 291,
    #         'prediction_results_file': f'./optimized/result/predicted_result_date_{date}.csv',
    #         'process_num': 32,
    #         'chunk_num': 5,
    #         'search_radius': 2000,
    #         'max_detour': 3000,
    #         'w_detour': 0,
    #         'w_pickup': 0,
    #         'w_shared': 0,
    #         "w_ride": -1,
    #         'pickup_time': 2,
    #         'speed': 600,  # m/min
    #         'max_iter_time': 100,
    #         "min_iter_time": 5,
    #         'M': 1e6,
    #         'epsilon': 1e-4,
    #         'n_v': 300,
    #         'beta': 1,
    #         'delta': 1,
    #         'K': 300,
    #         'OD_pickle_file': f'./result/different_date/0_shortest_path/tmp/OD_date_{date}.pickle',
    #         'graph_pickle_file': './tmp/graph.pickle',
    #         'shortest_path_pickle': f"./optimized/tmp/shortest_path_date_{date}.pickle",
    #         'ego_graph_pickle': f"./optimized/tmp/ego_graph_date_{date}.pickle",
    #         'match_csv': f"./optimized/tmp/match_date_{date}.csv",
    #         'node_pickle': "./tmp/node.pickle",
    #         'link_pickle': "./tmp/link.pickle",
    #         'node_pickle_file':'./tmp/node.pickle',
    #     }

    #     # 生成格式化的 setting.py 内容
    #     new_setting_content = "params = " + json.dumps(params, indent=4)

    #     # 写回 setting.py
    #     with open(setting_path, "w", encoding="utf-8") as f:
    #         f.write(new_setting_content)

    #     with open(params['OD_pickle_file'], 'rb') as f:
    #         OD_dict: dict = pickle.load(f)


    #     # 运行遗传算法并获取最优解
    #     best_theta, result = run_ga(OD_dict, params)
    #     end = time.time()
    #     print(f"Best solution for Date {date}: {best_theta}, time used {end - begin}")
    #     with open(f'./optimized/result/best_theta_date_{date}.pkl', 'wb') as f:
    #         pickle.dump(best_theta, f)
