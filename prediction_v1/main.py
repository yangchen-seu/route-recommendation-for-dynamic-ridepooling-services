import os
import json
import subprocess  # Import subprocess to execute commands sequentially
import time
setting_path = "settings.py"
files = os.listdir('./data/ODs/different_hour/')
start = time.time()

for file in files:
    print(f"Processing file: {file}")
    hour = int(file.split("_hour_")[1].split(".csv")[0])
    if hour < 12:
        continue 
    PROB = 0
    # 构造 params 字典
    params = {
        'lowest_road_class': 5,
        'max_combined_length': 1000,
        'OD_num': 291,
        'OD_file': f'./data/ODs/different_hour/{file}',
        'OD_passenger_num_file': f'./data/hourly_orders/{file[3:]}',
        'prediction_results_file': f'./result/different_hour/{PROB}_shortest_path/result/predicted_result_{hour}.csv',
        'date': f'{file[3:]}',
        'process_num': 32,
        'chunk_num': 5,
        'search_radius': 2000,
        'max_detour': 3000,
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
        'shortest_path_prob':PROB,
        'OD_pickle_file':f'./result/different_hour/{PROB}_shortest_path/tmp/OD_hour_{hour}.pickle',
        'graph_pickle_file':'./tmp/graph.pickle',
        'shortest_path_pickle':f"./result/different_hour/{PROB}_shortest_path/tmp/shortest_path_hour_{hour}.pickle",
        'ego_graph_pickle':f"./result/different_hour/{PROB}_shortest_path/tmp/ego_graph_hour_{hour}.pickle",
        'match_csv':f"./result/different_hour/{PROB}_shortest_path/tmp/match_hour_{hour}.csv",
        'node_pickle':"./tmp/node.pickle",
        'link_pickle':"./tmp/link.pickle",
        'node_pickle_file':f"./result/different_hour/{PROB}_shortest_path/tmp/node_hour_{hour}.pickle",
    }



    # 生成格式化的 setting.py 内容，保证 key 之间换行
    new_setting_content = "params = " + json.dumps(params, indent=4)

    # 写回 setting.py
    with open(setting_path, "w", encoding="utf-8") as f:
        f.write(new_setting_content)

    # 执行命令并等待完成
    print('generate pickle')
    subprocess.run(['python', 'generate_pickle.py'], check=True)

    print('shortest path')
    subprocess.run(['python', 'parallel_shortest_path_and_ego_graph.py'], check=True)

    print('matching pairs')
    subprocess.run(['python', 'parallel_searching_of_matching_pairs.py'], check=True)

    print('predict')
    subprocess.run(['python', 'predict.py'], check=True)


    PROB = 1
    # 构造 params 字典
    params = {
        'lowest_road_class': 5,
        'max_combined_length': 1000,
        'OD_num': 291,
        'OD_file': f'./data/ODs/different_hour/{file}',
        'OD_passenger_num_file': f'./data/hourly_orders/{file[3:]}',
        'prediction_results_file': f'./result/different_hour/{PROB}_shortest_path/result/predicted_result_{hour}.csv',
        'date': f'{file[3:]}',
        'process_num': 32,
        'chunk_num': 5,
        'search_radius': 2000,
        'max_detour': 3000,
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
        'shortest_path_prob':PROB,
        'OD_pickle_file':f'./result/different_hour/{PROB}_shortest_path/tmp/OD_hour_{hour}.pickle',
        'graph_pickle_file':'./tmp/graph.pickle',
        'shortest_path_pickle':f"./result/different_hour/{PROB}_shortest_path/tmp/shortest_path_hour_{hour}.pickle",
        'ego_graph_pickle':f"./result/different_hour/{PROB}_shortest_path/tmp/ego_graph_hour_{hour}.pickle",
        'match_csv':f"./result/different_hour/{PROB}_shortest_path/tmp/match_hour_{hour}.csv",
        'node_pickle':"./tmp/node.pickle",
        'link_pickle':"./tmp/link.pickle",
        'node_pickle_file':f"./result/different_hour/{PROB}_shortest_path/tmp/node_hour_{hour}.pickle",
    }



    # 生成格式化的 setting.py 内容，保证 key 之间换行
    new_setting_content = "params = " + json.dumps(params, indent=4)

    # 写回 setting.py
    with open(setting_path, "w", encoding="utf-8") as f:
        f.write(new_setting_content)

    # 执行命令并等待完成
    print('generate pickle')
    subprocess.run(['python', 'generate_pickle.py'], check=True)

    print('shortest path')
    subprocess.run(['python', 'parallel_shortest_path_and_ego_graph.py'], check=True)

    print('matching pairs')
    subprocess.run(['python', 'parallel_searching_of_matching_pairs.py'], check=True)

    print('predict')
    subprocess.run(['python', 'predict.py'], check=True)





###############different dates##################################


# setting_path = "settings.py"
# files = os.listdir('./data/ODs/different_date/')

# for file in files:
#     print(f"Processing file: {file}")
#     date = int(file.split("_hour_7_")[1].split(".csv")[0])
#     PROB = 1
#     # 构造 params 字典
#     params = {
#         'lowest_road_class': 5,
#         'max_combined_length': 1000,
#         'OD_num': 291,
#         'OD_file': f'./data/ODs/different_date/{file}',
#         'OD_passenger_num_file': f'./data/hourly_orders/2017-05_day_2_hour_7.csv',
#         'prediction_results_file': f'./result/different_date/{PROB}_shortest_path/result/predicted_result_{date}.csv',
#         'date': date,
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
#         'shortest_path_prob':PROB,
#         'OD_pickle_file':f'./result/different_date/{PROB}_shortest_path/tmp/OD_date_{date}.pickle',
#         'shortest_path_pickle':f"./result/different_date/{PROB}_shortest_path/tmp/shortest_path_date_{date}.pickle",
#         'ego_graph_pickle':f"./result/different_date/{PROB}_shortest_path/tmp/ego_graph_date_{date}.pickle",
#         'match_csv':f"./result/different_date/{PROB}_shortest_path/tmp/match_date_{date}.csv",
#         'node_pickle':"./tmp/node.pickle",
#         'link_pickle':"./tmp/link.pickle",
#         'graph_pickle_file':'./tmp/graph.pickle',
#         'node_pickle_file':f"./result/different_date/{PROB}_shortest_path/tmp/node_date_{date}.pickle",
#     }

#     # 生成格式化的 setting.py 内容，保证 key 之间换行
#     new_setting_content = "params = " + json.dumps(params, indent=4)

#     # 写回 setting.py
#     with open(setting_path, "w", encoding="utf-8") as f:
#         f.write(new_setting_content)

#     # 执行命令并等待完成
#     print('generate pickle')
#     subprocess.run(['python', 'generate_pickle.py'], check=True)

#     print('shortest path')
#     subprocess.run(['python', 'parallel_shortest_path_and_ego_graph.py'], check=True)

#     print('matching pairs')
#     subprocess.run(['python', 'parallel_searching_of_matching_pairs.py'], check=True)

#     print('predict')
#     subprocess.run(['python', 'predict.py'], check=True)







#     PROB = 0
#     # 构造 params 字典
#     params = {
#         'lowest_road_class': 5,
#         'max_combined_length': 1000,
#         'OD_num': 291,
#         'OD_file': f'./data/ODs/different_date/{file}',
#         'OD_passenger_num_file': f'./data/hourly_orders/2017-05_day_2_hour_7.csv',
#         'prediction_results_file': f'./result/different_date/{PROB}_shortest_path/result/predicted_result_{date}.csv',
#         'date': date,
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
#         'shortest_path_prob':PROB,
#         'OD_pickle_file':f'./result/different_date/{PROB}_shortest_path/tmp/OD_date_{date}.pickle',
#         'shortest_path_pickle':f"./result/different_date/{PROB}_shortest_path/tmp/shortest_path_date_{date}.pickle",
#         'ego_graph_pickle':f"./result/different_date/{PROB}_shortest_path/tmp/ego_graph_date_{date}.pickle",
#         'match_csv':f"./result/different_date/{PROB}_shortest_path/tmp/match_date_{date}.csv",
#         'node_pickle':"./tmp/node.pickle",
#         'link_pickle':"./tmp/link.pickle",
#         'graph_pickle_file':'./tmp/graph.pickle',
#         'node_pickle_file':f"./result/different_date/{PROB}_shortest_path/tmp/node_date_{date}.pickle",
#     }

#     # 生成格式化的 setting.py 内容，保证 key 之间换行
#     new_setting_content = "params = " + json.dumps(params, indent=4)

#     # 写回 setting.py
#     with open(setting_path, "w", encoding="utf-8") as f:
#         f.write(new_setting_content)

#     # 执行命令并等待完成
#     print('generate pickle')
#     subprocess.run(['python', 'generate_pickle.py'], check=True)

#     print('shortest path')
#     subprocess.run(['python', 'parallel_shortest_path_and_ego_graph.py'], check=True)

#     print('matching pairs')
#     subprocess.run(['python', 'parallel_searching_of_matching_pairs.py'], check=True)

#     print('predict')
#     subprocess.run(['python', 'predict.py'], check=True)



# end = time.time()
# print("All tasks completed! time:{}".format(end-start))
