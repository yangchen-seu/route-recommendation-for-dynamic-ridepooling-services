import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
from zoopt import Dimension, Objective, Parameter, ExpOpt, Opt
import Predict_class

# 读取数据
pre = Predict_class.Predict()
with open("tmp/OD.pickle", 'rb') as f:
    OD_dict: dict = pickle.load(f)
with open("tmp/node.pickle", 'rb') as f:
    node_dict: dict = pickle.load(f)

start = time.time()

print('stage 1')
# 变量数量
num_ODs = len(OD_dict)
num_nodes = len(node_dict)
num_dimensions = num_ODs
# num_dimensions = num_ODs + num_nodes * num_ODs

def generate_random_numbers(length):
    """生成归一化随机数数组，使其加和为 1"""
    random_numbers = np.random.rand(length)
    return random_numbers / random_numbers.sum()

print('stage 2')
# 目标函数
def objective_function(solution):
    params = solution.get_x()
    theta_values = params[:num_ODs]  # Theta 参数
    # pi_values = params[num_ODs:]     # Pi 参数

    # 构造 Theta
    theta_dict = {OD_id: {'shortest_path': theta_values[i], 'highest_potential_path': 1 - theta_values[i]} for i, OD_id in enumerate(OD_dict.keys())}
    
    # 构造 Pi
    # pi_dict = {}
    # index = 0
    # for node_id in node_dict.keys():
    #     pi_dict[node_id] = pi_values[index:index + num_ODs]
    #     index += num_ODs
    # print('stage 3')
    # 计算平台利润
    start = time.time()
    # platform_profit = pre.run(pi_dict, theta_dict)
    platform_profit = pre.run(theta_dict)
    end = time.time()
    # print('equilibrium time', end - start)
    
    return -platform_profit  # 负利润（最小化问题）

# 定义搜索空间
dim = Dimension(num_dimensions, [[0, 1]] * num_dimensions, [True] * num_dimensions)  # 每个维度在 [0,1] 内
objective = Objective(objective_function, dim)
print('stage 4')
# 并行优化
import multiprocessing
cpu_cores = multiprocessing.cpu_count()  # 获取 CPU 核心数
solution_opt = Opt.min(objective, Parameter(budget=10000 , parallel=True, server_num=128))  # * num_dimensions
print('stage 5')
# 获取最优解
best_solution = solution_opt.get_x()
best_function_value = solution_opt.get_value()
history_value = objective.get_history_bestsofar()

print("Best parameters: {}".format(best_solution))
print("Best function value: {}".format(best_function_value))

# 绘制优化过程
plt.plot(history_value)
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Optimization Progress')
plt.show()
plt.savefig('result/SRACO_iteration_10000.png')


end = time.time()
print('time',end - start)