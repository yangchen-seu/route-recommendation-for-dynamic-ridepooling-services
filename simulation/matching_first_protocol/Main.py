
import Simulation as sm
import Config
import pandas as pd
import numpy as np


class Main():

    def __init__(self,cfg , prob = 0) -> None:
        self.simu = sm.Simulation(cfg, prob)
        self.test_episodes = 1
 
 
    def run(self):
        while True:
            results, done = self.simu.step()
            if done:
                break



        # print('waitingTime:',results['waitingTime'])
        # print('detour_distance:',results['detour_distance'])
        # print('pickup_time:',results['pickup_time'])
        # print('shared_distance:',results['shared_distance'])
        # print('total_ride_distance:',results['total_ride_distance'])
        # print('saved_ride_distance:',results['saved_ride_distance'])
        # print('mean_ride_distance:',results['mean_ride_distance'])
        # print('mean_saved_ride_distance:',results['mean_saved_ride_distance'])
        # print('platform_income:',results['platform_income'])
        # print('response_rate:',results['response_rate'])
        # print('carpool_rate:',results['carpool_rate'])

        
        # self.plot_metrics(self.simu.cfg, metric = self.simu.detour_distance, metric_name = 'detour_distance')
        # self.plot_metrics(self.simu.cfg, metric =self.simu.traveltime, metric_name = 'traveltime')
        # self.plot_metrics(self.simu.cfg, metric = self.simu.waitingtime, metric_name = 'waiting_time')
        # self.plot_metrics(self.simu.cfg, metric = self.simu.pickup_time, metric_name = 'pickup_time')
        # self.plot_metrics(self.simu.cfg, metric = self.simu.platform_income, metric_name = 'platform_income')
        # self.plot_metrics(self.simu.cfg, metric = self.simu.shared_distance, metric_name = 'shared_distance')
 
        return results

     # 绘图
    # def plot_metrics(self,cfg, metric,  metric_name, algo = 'batchmatching', env_name='ridesharing'):
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
    #     sns.set()
    #     plt.figure() # 创建一个图形实例，方便同时多画几个图
    #     plt.title("{} of {} for {}".format(metric_name, algo,env_name))
    #     plt.xlabel('orders')
    #     plt.plot(metric,label=metric_name)
    #     plt.legend()
    #     plt.savefig('figure/{}.png'.format( metric_name))       



def different_date(prob = 0):
    import os
    files = os.listdir('input/different_date/orders/')
    dic = {}
    import pickle

    with open('output/different_date_{}_shortest_path_results.pkl'.format(prob), "wb") as tf:
        for file in files:
            print(file)
            cfg = Config.Config()
            cfg.progress_target = False 
            # print('ratio:',cfg.order_driver_ratio)
            cfg.date = file.split('.')[0]
            cfg.order_file = 'input/different_date/orders/'+ file
            cfg.simulation_begin_time = ' 07:00:00' # 仿真开始时间
            cfg.simulation_end_time = ' 08:00:00' # 仿真结束时间
            cfg.order_driver_ratio = 4

            cfg.OD_pickle_path = f'/root/rent/yangchen/route_recommendation/prediction_v3/result/different_date/{prob}_shortest_path/tmp/OD_date_{int(cfg.date[-2:])}.pickle'
            print('date', int(cfg.date[-2:]) )
            cfg.output_path = 'output/'
            cfg.graph_pickle_file = './input/graph.pickle'
            import time
            start = time.time()
            ma = Main(cfg, prob)
            res = ma.run() 
            end = time.time()
            print('file:{},执行时间:{}'.format(file, end - start))
            dic[file.split('.')[0]] = res

            result = pd.DataFrame.from_dict(res, orient='index').loc[:, [
            "O_location", "D_location", "his_order_num", "responded_order_num", "pooled_order_num",
            "waitingtime", "detour_distance", "pickup_time", "shared_distance",
            "total_travel_distance", "saved_travel_distance"]]


            # 计算新增字段
            result["response_rate"] = result["responded_order_num"] / result["his_order_num"]
            result["pooling_rate"] = result["pooled_order_num"] / result["responded_order_num"].replace(0, np.nan)  # 避免除零错误
            result["avg_detour_distance"] = result["detour_distance"].apply(lambda x: np.mean(x) if x else np.nan)
            result["avg_pickup_distance"] = result["pickup_time"].apply(lambda x: np.mean(x) if x else np.nan)
            result["avg_shared_distance"] = result["shared_distance"].apply(lambda x: np.mean(x) if x else np.nan)
            result["avg_total_travel_distance"] = result["total_travel_distance"].apply(lambda x: np.mean(x) if x else np.nan)
            result["avg_saved_travel_distance"] = result["saved_travel_distance"].apply(lambda x: np.mean(x) if x else np.nan)

            # 填充 NaN 值为 0（可以按需调整）
            result.fillna(0, inplace=True)

            # 保存新的数据文件
            result.index.name = "OD_id"
            
            # 计算全局统计指标
            overall_response_rate = result["responded_order_num"].sum() / result["his_order_num"].sum()
            overall_pooling_rate = result["pooled_order_num"].sum() / result["responded_order_num"].replace(0, np.nan).sum()

            # 计算平均值（排除空列表）
            overall_avg_waiting_time = np.mean([item for sublist in result["waitingtime"] for item in sublist])
            overall_avg_detour_distance = np.mean([item for sublist in result["detour_distance"] for item in sublist])
            overall_avg_pickup_time = np.mean([item for sublist in result["pickup_time"] for item in sublist])
            overall_avg_shared_distance = np.mean([item for sublist in result["shared_distance"] for item in sublist])
            overall_avg_total_travel_distance = np.mean([item for sublist in result["total_travel_distance"] for item in sublist])
            overall_avg_saved_travel_distance = np.mean([item for sublist in result["saved_travel_distance"] for item in sublist])

            # 打印结果
            print("全局应答率:", overall_response_rate)
            print("全局拼成率:", overall_pooling_rate)
            print("全局平均等待时间:", overall_avg_waiting_time)
            print("全局平均绕行距离:", overall_avg_detour_distance)
            print("全局平均接驾时间:", overall_avg_pickup_time)
            print("全局平均共享距离:", overall_avg_shared_distance)
            print("全局平均总行程距离:", overall_avg_total_travel_distance)
            print("全局平均节省的行程距离:", overall_avg_saved_travel_distance)
            print("全局总计节省的行程距离:", np.sum([item for sublist in result["saved_travel_distance"] for item in sublist]))

            result.to_csv(f'output/different_date/{prob}_shortest_path/simulation_date_{int(cfg.date[-2:])}.csv')
            
        pickle.dump(dic, tf)

    return 


def different_hour( prob=0):
    """
    针对不同小时的拼车仿真，计算各小时的拼车成功率等统计指标
    :param date: 指定的日期，例如 "2017_05_day_02"
    :param prob: 影响匹配的概率参数
    """
    
    input_path = f'input/different_hour/orders/'
    import os
    import time
    import pickle
    files = os.listdir('input/different_hour/orders/')
    results_dict = {}
    
    output_pickle_path = f'output/different_hour_{prob}_shortest_path_results.pkl'
    
    with open(output_pickle_path, "wb") as tf:
        for file in files:
            print(f"Processing file: {file}")
            
            cfg = Config.Config()
            cfg.order_driver_ratio = 4
            cfg.progress_target = False 

            cfg.order_file = os.path.join(input_path, file)
            hour = int(file.split("hour_")[1].split(".csv")[0])  # 提取小时
            cfg.date="2017-05-02"
            cfg.simulation_begin_time = f' {hour:02d}:00:00'  # 开始时间
            if hour == 23:
                cfg.simulation_end_time = f' {hour:02d}:59:59'  # 结束时间
            else:
                cfg.simulation_end_time = f' {hour+1:02d}:00:00'  # 结束时间
            
            if hour == 5:
                continue
            if prob == 0:
                cfg.detour_distance_threshold = 1100

            cfg.graph_pickle_file = './input/graph.pickle'
            cfg.OD_pickle_path = f'/root/rent/yangchen/route_recommendation/prediction_v3/result/different_hour/{prob}_shortest_path/tmp/OD_hour_{hour}.pickle'
            cfg.output_path = 'output/'
            
            start_time = time.time()
            ma = Main(cfg, prob)
            res = ma.run()
            end_time = time.time()
            print(f'File: {file}, Execution Time: {end_time - start_time:.2f} seconds')
            
            results_dict[file.split('.')[0]] = res
            result_df = pd.DataFrame.from_dict(res, orient='index').loc[:, [
                "O_location", "D_location", "his_order_num", "responded_order_num", "pooled_order_num",
                "waitingtime", "detour_distance", "pickup_time", "shared_distance",
                "total_travel_distance", "saved_travel_distance"]]

            # 计算额外统计指标
            result_df["response_rate"] = result_df["responded_order_num"] / result_df["his_order_num"]
            result_df["pooling_rate"] = result_df["pooled_order_num"] / result_df["responded_order_num"].replace(0, np.nan)
            
            for col in ["detour_distance", "pickup_time", "shared_distance", "total_travel_distance", "saved_travel_distance"]:
                result_df[f"avg_{col}"] = result_df[col].apply(lambda x: np.mean(x) if x else np.nan)
            
            result_df.fillna(0, inplace=True)
            result_df.index.name = "OD_id"
            
            # 计算全局统计量
            overall_response_rate = result_df["responded_order_num"].sum() / result_df["his_order_num"].sum()
            overall_pooling_rate = result_df["pooled_order_num"].sum() / result_df["responded_order_num"].replace(0, np.nan).sum()
            
            overall_avg_stats = {}
            for col in ["waitingtime", "detour_distance", "pickup_time", "shared_distance", "total_travel_distance", "saved_travel_distance"]:
                overall_avg_stats[col] = np.mean([item for sublist in result_df[col] for item in sublist])
            
            # 打印全局统计结果
            print(f"Hour: {hour}")
            print(f"Overall Response Rate: {overall_response_rate:.4f}")
            print(f"Overall Pooling Rate: {overall_pooling_rate:.4f}")
            for key, value in overall_avg_stats.items():
                print(f"Overall Avg {key.replace('_', ' ').title()}: {value:.2f}")
            print(f"Total Saved Travel Distance: {np.sum([item for sublist in result_df['saved_travel_distance'] for item in sublist]):.2f}")
            print("-" * 50)


            result_df.to_csv(f'output/different_hour/{prob}_shortest_path/simulation_hour_{hour}.csv')

        pickle.dump(results_dict, tf)
    
    print("All hourly simulations completed!")



def one_day_test(prob):
    for detour in  [1300,1400,1500,1600,1700,1800,1900,2000]:
        cfg = Config.Config()
        cfg.detour_distance_threshold=detour
        cfg.order_driver_ratio = 4
        cfg.order_file = 'input/different_hour/orders/hour_07.csv'
        cfg.date = '2017-05-02'
        cfg.simulation_begin_time = ' 07:00:00' # 仿真开始时间
        cfg.simulation_end_time = ' 08:00:00' # 仿真结束时间
        cfg.demand_ratio = 0.25
        cfg.OD_pickle_path = 'input/different_hour/0_shortest_path/tmp/OD_hour_7.pickle'
        cfg.OD_pickle_path = f'/root/rent/yangchen/route_recommendation/prediction_v3/result/different_hour/{prob}_shortest_path/tmp/OD_hour_7.pickle'
        cfg.graph_pickle_file = './input/graph.pickle'

        cfg.progress_target = False
        print('ratio:',cfg.order_driver_ratio)
        import time
        start = time.time()
        ma = Main(cfg, prob)
        res = ma.run() 
        end = time.time()
        print('执行时间{},order_driver_ratio:{}'.format(end - start, cfg.order_driver_ratio))
        print('ratio:',ma.simu.vehicle_num / len(ma.simu.order_list))
        result = pd.DataFrame.from_dict(res, orient='index').loc[:, [
        "O_location", "D_location", "his_order_num", "responded_order_num", "pooled_order_num",
        "waitingtime", "detour_distance", "pickup_time", "shared_distance",
        "total_travel_distance", "saved_travel_distance"]]


        # 计算新增字段
        result["response_rate"] = result["responded_order_num"] / result["his_order_num"]
        result["pooling_rate"] = result["pooled_order_num"] / result["responded_order_num"].replace(0, np.nan)  # 避免除零错误
        result["avg_detour_distance"] = result["detour_distance"].apply(lambda x: np.mean(x) if x else np.nan)
        result["avg_pickup_distance"] = result["pickup_time"].apply(lambda x: np.mean(x) if x else np.nan)
        result["avg_shared_distance"] = result["shared_distance"].apply(lambda x: np.mean(x) if x else np.nan)
        result["avg_total_travel_distance"] = result["total_travel_distance"].apply(lambda x: np.mean(x) if x else np.nan)
        result["avg_saved_travel_distance"] = result["saved_travel_distance"].apply(lambda x: np.mean(x) if x else np.nan)

        # 填充 NaN 值为 0（可以按需调整）
        result.fillna(0, inplace=True)

        # 保存新的数据文件
        result.index.name = "OD_id"
        # result.to_csv("simulation_result_with_stats_{}_shortest_path.csv".format(prob))
        
        # 计算全局统计指标
        overall_response_rate = result["responded_order_num"].sum() / result["his_order_num"].sum()
        overall_pooling_rate = result["pooled_order_num"].sum() / result["responded_order_num"].replace(0, np.nan).sum()

        # 计算平均值（排除空列表）
        overall_avg_waiting_time = np.mean([item for sublist in result["waitingtime"] for item in sublist])
        overall_avg_detour_distance = np.mean([item for sublist in result["detour_distance"] for item in sublist])
        overall_avg_pickup_time = np.mean([item for sublist in result["pickup_time"] for item in sublist])
        overall_avg_shared_distance = np.mean([item for sublist in result["shared_distance"] for item in sublist])
        overall_avg_total_travel_distance = np.mean([item for sublist in result["total_travel_distance"] for item in sublist])
        overall_avg_saved_travel_distance = np.mean([item for sublist in result["saved_travel_distance"] for item in sublist])

        # 打印结果
        print('detour',detour)
        print("全局应答率:", overall_response_rate)
        print("全局拼成率:", overall_pooling_rate)
        print("全局平均等待时间:", overall_avg_waiting_time)
        print("全局平均绕行距离:", overall_avg_detour_distance)
        print("全局平均接驾时间:", overall_avg_pickup_time)
        print("全局平均共享距离:", overall_avg_shared_distance)
        print("全局平均总行程距离:", overall_avg_total_travel_distance)
        print("全局平均节省的行程距离:", overall_avg_saved_travel_distance)
        print("全局总计节省的行程距离:", np.sum([item for sublist in result["saved_travel_distance"] for item in sublist]))








def different_date_optimize():
    import os
    files = os.listdir('input/different_date/orders/')
    dic = {}
    import pickle

    with open('./input/best_solutions.pkl', "rb") as f:
        best_solutions = pickle.load(f)

    with open('output/different_date_optimize_results.pkl', "wb") as tf:
        for file in files:
            print(file)
            cfg = Config.Config()
            cfg.progress_target = False 
            # print('ratio:',cfg.order_driver_ratio)
            cfg.date = file.split('.')[0]
            cfg.order_file = 'input/different_date/orders/'+ file
            cfg.simulation_begin_time = ' 07:00:00' # 仿真开始时间
            cfg.simulation_end_time = ' 08:00:00' # 仿真结束时间
            cfg.OD_pickle_path = f'/root/rent/yangchen/route_recommendation/prediction_v3/result/different_date/0_shortest_path/tmp/OD_date_{int(cfg.date[-2:])}.pickle'
            print('date', int(cfg.date[-2:]) )
            cfg.output_path = 'output/'
            cfg.graph_pickle_file = './input/graph.pickle'

            import time
            # cfg.Theta = best_solutions[int(cfg.date[-2:])]
            # cfg.Theta = best_solutions[hour]
            with open(f'/root/rent/yangchen/route_recommendation/prediction_v3/optimized/result/best_theta_date_{int(cfg.date[-2:])}.pkl','rb') as f:
                cfg.Theta = pickle.load(f)

            cfg.optimize_target = True
            cfg.order_driver_ratio = 4
            start = time.time()
            ma = Main(cfg, prob=0)
            res = ma.run() 
            end = time.time()
            print('file:{},执行时间:{}'.format(file, end - start))
            dic[file.split('.')[0]] = res

            result = pd.DataFrame.from_dict(res, orient='index').loc[:, [
            "O_location", "D_location", "his_order_num", "responded_order_num", "pooled_order_num",
            "waitingtime", "detour_distance", "pickup_time", "shared_distance",
            "total_travel_distance", "saved_travel_distance"]]


            # 计算新增字段
            result["response_rate"] = result["responded_order_num"] / result["his_order_num"]
            result["pooling_rate"] = result["pooled_order_num"] / result["responded_order_num"].replace(0, np.nan)  # 避免除零错误
            result["avg_detour_distance"] = result["detour_distance"].apply(lambda x: np.mean(x) if x else np.nan)
            result["avg_pickup_distance"] = result["pickup_time"].apply(lambda x: np.mean(x) if x else np.nan)
            result["avg_shared_distance"] = result["shared_distance"].apply(lambda x: np.mean(x) if x else np.nan)
            result["avg_total_travel_distance"] = result["total_travel_distance"].apply(lambda x: np.mean(x) if x else np.nan)
            result["avg_saved_travel_distance"] = result["saved_travel_distance"].apply(lambda x: np.mean(x) if x else np.nan)

            # 填充 NaN 值为 0（可以按需调整）
            result.fillna(0, inplace=True)

            # 保存新的数据文件
            result.index.name = "OD_id"
            
            # 计算全局统计指标
            overall_response_rate = result["responded_order_num"].sum() / result["his_order_num"].sum()
            overall_pooling_rate = result["pooled_order_num"].sum() / result["responded_order_num"].replace(0, np.nan).sum()

            # 计算平均值（排除空列表）
            overall_avg_waiting_time = np.mean([item for sublist in result["waitingtime"] for item in sublist])
            overall_avg_detour_distance = np.mean([item for sublist in result["detour_distance"] for item in sublist])
            overall_avg_pickup_time = np.mean([item for sublist in result["pickup_time"] for item in sublist])
            overall_avg_shared_distance = np.mean([item for sublist in result["shared_distance"] for item in sublist])
            overall_avg_total_travel_distance = np.mean([item for sublist in result["total_travel_distance"] for item in sublist])
            overall_avg_saved_travel_distance = np.mean([item for sublist in result["saved_travel_distance"] for item in sublist])

            # 打印结果
            print("全局应答率:", overall_response_rate)
            print("全局拼成率:", overall_pooling_rate)
            print("全局平均等待时间:", overall_avg_waiting_time)
            print("全局平均绕行距离:", overall_avg_detour_distance)
            print("全局平均接驾时间:", overall_avg_pickup_time)
            print("全局平均共享距离:", overall_avg_shared_distance)
            print("全局平均总行程距离:", overall_avg_total_travel_distance)
            print("全局平均节省的行程距离:", overall_avg_saved_travel_distance)
            print("全局总计节省的行程距离:", np.sum([item for sublist in result["saved_travel_distance"] for item in sublist]))

            result.to_csv(f'output/different_date/optimize/simulation_date_{int(cfg.date[-2:])}.csv')
            
        pickle.dump(dic, tf)

    return 




def different_hour_optimize():
    import os
    files = os.listdir('input/different_hour/orders/')
    dic = {}
    import pickle

    with open('./input/best_solutions_different_hour.pkl', "rb") as f:
        best_solutions = pickle.load(f)

    with open('output/different_hour_optimize_results.pkl', "wb") as tf:
        for file in files:
            print(f"Processing file: {file}")
            
            cfg = Config.Config()
            cfg.order_driver_ratio = 4
            cfg.detour_distance_threshold = 1200
            cfg.progress_target = False 
            cfg.date = '2017-05-02'
            cfg.order_file = os.path.join('input/different_hour/orders/', file)
            hour = int(file.split("hour_")[1].split(".csv")[0])  # 提取小时
            
            if hour == 5:
                continue

            cfg.simulation_begin_time = f' {hour:02d}:00:00'  # 开始时间
            if hour == 23:
                cfg.simulation_end_time = f' {hour:02d}:59:59'  # 结束时间
            else:
                cfg.simulation_end_time = f' {hour+1:02d}:00:00'  # 结束时间
            

            cfg.OD_pickle_path = f'/root/rent/yangchen/route_recommendation/prediction_v3/result/different_hour/0_shortest_path/tmp/OD_hour_{hour}.pickle'

            cfg.output_path = 'output/'
            cfg.graph_pickle_file = './input/graph.pickle'


            import time
            # cfg.Theta = best_solutions[hour]
            with open(f'/root/rent/yangchen/route_recommendation/prediction_v3/optimized/result/best_theta_hour_{hour}.pkl','rb') as f:
                cfg.Theta = pickle.load(f)

            cfg.optimize_target = True
            start = time.time()
            ma = Main(cfg, prob=0)
            res = ma.run() 
            end = time.time()
            print('file:{},执行时间:{}'.format(file, end - start))
            dic[file.split('.')[0]] = res

            result = pd.DataFrame.from_dict(res, orient='index').loc[:, [
            "O_location", "D_location", "his_order_num", "responded_order_num", "pooled_order_num",
            "waitingtime", "detour_distance", "pickup_time", "shared_distance",
            "total_travel_distance", "saved_travel_distance"]]


            # 计算新增字段
            result["response_rate"] = result["responded_order_num"] / result["his_order_num"]
            result["pooling_rate"] = result["pooled_order_num"] / result["responded_order_num"].replace(0, np.nan)  # 避免除零错误
            result["avg_detour_distance"] = result["detour_distance"].apply(lambda x: np.mean(x) if x else np.nan)
            result["avg_pickup_distance"] = result["pickup_time"].apply(lambda x: np.mean(x) if x else np.nan)
            result["avg_shared_distance"] = result["shared_distance"].apply(lambda x: np.mean(x) if x else np.nan)
            result["avg_total_travel_distance"] = result["total_travel_distance"].apply(lambda x: np.mean(x) if x else np.nan)
            result["avg_saved_travel_distance"] = result["saved_travel_distance"].apply(lambda x: np.mean(x) if x else np.nan)

            # 填充 NaN 值为 0（可以按需调整）
            result.fillna(0, inplace=True)

            # 保存新的数据文件
            result.index.name = "OD_id"
            
            # 计算全局统计指标
            overall_response_rate = result["responded_order_num"].sum() / result["his_order_num"].sum()
            overall_pooling_rate = result["pooled_order_num"].sum() / result["responded_order_num"].replace(0, np.nan).sum()

            # 计算平均值（排除空列表）
            overall_avg_waiting_time = np.mean([item for sublist in result["waitingtime"] for item in sublist])
            overall_avg_detour_distance = np.mean([item for sublist in result["detour_distance"] for item in sublist])
            overall_avg_pickup_time = np.mean([item for sublist in result["pickup_time"] for item in sublist])
            overall_avg_shared_distance = np.mean([item for sublist in result["shared_distance"] for item in sublist])
            overall_avg_total_travel_distance = np.mean([item for sublist in result["total_travel_distance"] for item in sublist])
            overall_avg_saved_travel_distance = np.mean([item for sublist in result["saved_travel_distance"] for item in sublist])

            # 打印结果
            print("全局应答率:", overall_response_rate)
            print("全局拼成率:", overall_pooling_rate)
            print("全局平均等待时间:", overall_avg_waiting_time)
            print("全局平均绕行距离:", overall_avg_detour_distance)
            print("全局平均接驾时间:", overall_avg_pickup_time)
            print("全局平均共享距离:", overall_avg_shared_distance)
            print("全局平均总行程距离:", overall_avg_total_travel_distance)
            print("全局平均节省的行程距离:", overall_avg_saved_travel_distance)
            print("全局总计节省的行程距离:", np.sum([item for sublist in result["saved_travel_distance"] for item in sublist]))

            result.to_csv(f'output/different_hour/optimize/simulation_hour_{hour}.csv')
            
        pickle.dump(dic, tf)

    return 

one_day_test(prob = 0)
# one_day_test(prob = 1)
# different_date(prob = 0)
# different_date(prob = 1)


# different_date_optimize()

# different_hour(prob = 0)
# different_hour(prob = 1)
# different_hour_optimize()
# different_ratio()