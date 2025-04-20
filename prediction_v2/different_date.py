import os
import json
import subprocess  # Import subprocess to execute commands sequentially

setting_path = "settings.py"
files = os.listdir('./data/ODs/2017_05_date_02/')

for file in files:
    print(f"Processing file: {file}")

    # 构造 params 字典
    params = {
        'lowest_road_class': 5,
        'max_combined_length': 1000,
        'OD_num': 291,
        'OD_file': f'./data/ODs/2017_05_date_02/{file}',
        'OD_passenger_num_file': f'./data/hourly_orders/{file[3:]}',
        'prediction_results_file': './result/2017_05_date_02/',
        'date': f'{file[3:]}',
        'process_num': 4,
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
        'shortest_path_prob':0,
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

    # # 处理预测结果
    # os.rename('result/predict_result.csv', f'result/predict_result_{file}')

print("All tasks completed!")




setting_path = "settings.py"
files = os.listdir('./data/ODs/2017_05_hour_07/')

for file in files:
    print(f"Processing file: {file}")

    # 构造 params 字典
    params = {
        'lowest_road_class': 5,
        'max_combined_length': 1000,
        'OD_num': 291,
        'OD_file': f'./data/ODs/2017_05_hour_07/{file}',
        'OD_passenger_num_file': f'./data/hourly_orders/2017-05_day_2_hour_7.csv',
        'prediction_results_file': './result/2017_05_hour_07/',
        'date': f'{file[3:]}',
        'process_num': 4,
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
        'shortest_path_prob':0,
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

    # # 处理预测结果
    # os.rename('result/predict_result.csv', f'result/predict_result_{file}')

print("All tasks completed!")






###########################################################################



import os
import json
import subprocess  # Import subprocess to execute commands sequentially

setting_path = "settings.py"
files = os.listdir('./data/ODs/2017_05_date_02/')

for file in files:
    print(f"Processing file: {file}")

    # 构造 params 字典
    params = {
        'lowest_road_class': 5,
        'max_combined_length': 1000,
        'OD_num': 291,
        'OD_file': f'./data/ODs/2017_05_date_02/{file}',
        'OD_passenger_num_file': f'./data/hourly_orders/{file[3:]}',
        'prediction_results_file': './result/2017_05_date_02/',
        'date': f'{file[3:]}',
        'process_num': 4,
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
        'shortest_path_prob':1,
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

    # # 处理预测结果
    # os.rename('result/predict_result.csv', f'result/predict_result_{file}')

print("All tasks completed!")




setting_path = "settings.py"
files = os.listdir('./data/ODs/2017_05_hour_07/')

for file in files:
    print(f"Processing file: {file}")

    # 构造 params 字典
    params = {
        'lowest_road_class': 5,
        'max_combined_length': 1000,
        'OD_num': 291,
        'OD_file': f'./data/ODs/2017_05_hour_07/{file}',
        'OD_passenger_num_file': f'./data/hourly_orders/2017-05_day_2_hour_7.csv',
        'prediction_results_file': './result/2017_05_hour_07/',
        'date': f'{file[3:]}',
        'process_num': 4,
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
        'shortest_path_prob':1,
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

    # # 处理预测结果
    # os.rename('result/predict_result.csv', f'result/predict_result_{file}')

print("All tasks completed!")
