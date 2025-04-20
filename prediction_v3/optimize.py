import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
from zoopt import Dimension, Objective, Parameter, Opt, Solution
import Predict_class
import json
import os
import pandas as pd


def run_zoopt(OD_dict, params):
    begin = time.time()
    pre = Predict_class.Predict()
    num_ODs = len(OD_dict)
    num_dimensions = num_ODs  # 只优化 theta 参数

    def objective_function(solution):
        theta_values = solution.get_x()
        theta_dict = {OD_id: {'shortest_path': theta_values[i], 'highest_potential_path': 1 - theta_values[i]} 
                      for i, OD_id in enumerate(OD_dict.keys())}
        try:
            platform_profit, result = pre.run(theta_dict, params)
        except Exception as e:
            print(f"Error in pre.run(): {e}")
            return float('inf')  # 避免影响优化
        return -platform_profit  # 负利润（最小化问题）

    # 定义搜索空间
    dim = Dimension(num_dimensions, [[0,0.2,0.4,0.6,0.8 ,1]] * num_dimensions, [False] * num_dimensions)
    objective = Objective(objective_function, dim)
    
    # 并行优化
    sol1 = Solution(x=[0] * num_dimensions)
    sol2 = Solution(x=[1] * num_dimensions)
    parameter = Parameter(budget=5000, low_dimension=dim, parallel=True, server_num=128,init_samples=[sol1, sol2]) # 
    
    solution_opt = Opt.min(objective, parameter )
    
    best_solution = solution_opt.get_x()
    best_function_value = solution_opt.get_value()
    history_value = objective.get_history_bestsofar()
    
    print("Best parameters:", best_solution)
    print("Best function value:", best_function_value)
    end = time.time()

    print('time:',end - begin)
    return best_solution, history_value


if __name__ == '__main__':
    setting_path = "settings.py"
    hours = [i for i in range(6, 24)]
    os.makedirs("result", exist_ok=True)
    
    for hour in hours:
        print(f"Processing hour: {hour}")
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
            'OD_pickle_file': f'./optimized/tmp/OD_hour_{hour}.pickle',
            'graph_pickle_file': './tmp/graph.pickle',
            'shortest_path_pickle': f"./optimized/tmp/shortest_path_hour_{hour}.pickle",
            'ego_graph_pickle': f"./optimized/tmp/ego_graph_hour_{hour}.pickle",
            'match_csv': f"./optimized/tmp/match_hour_{hour}.csv",
            'node_pickle': "./tmp/node.pickle",
            'link_pickle': "./tmp/link.pickle",
            'node_pickle_file':'./tmp/node.pickle',
        }
        
        with open(params['OD_pickle_file'], 'rb') as f:
            OD_dict = pickle.load(f)
        
        best_theta, history = run_zoopt(OD_dict, params)
        
        # 保存结果
        with open(f'./optimized/result/best_theta_hour_{hour}.pkl', 'wb') as f:
            pickle.dump(best_theta, f)
        
        plt.plot(history)
        plt.xlabel('Iteration')
        plt.ylabel('Objective Function Value')
        plt.title(f'ZOOpt Optimization Progress (Hour {hour})')
        plt.savefig(f'result/ZOOpt_Optimization_Hour_{hour}.png')
        plt.close()
