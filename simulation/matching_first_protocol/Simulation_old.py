
import time
import pandas as pd
import numpy as np
import Network as net
import Seeker
import random
import pickle
import Vehicle
import os
import networkx as nx

class Simulation():
    # 输入不同的订单，输出系统的派单结果，以及各种指标
    def __init__(self, cfg , prob = 0.5) -> None:
        self.cfg = cfg
        self.date = cfg.date
        self.prob = prob
        print('self.prob',self.prob)
        self.order_list = pd.read_csv(self.cfg.order_file)
        self.vehicle_num = int(len(self.order_list) / cfg.order_driver_ratio )
        self.order_list['beginTime_stamp'] = self.order_list['dwv_order_make_haikou_1.departure_time'].apply(
            lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
        self.begin_time = time.mktime(time.strptime(
            cfg.date + cfg.simulation_begin_time, "%Y-%m-%d %H:%M:%S"))
        self.end_time = time.mktime(time.strptime(
            cfg.date + cfg.simulation_end_time, "%Y-%m-%d %H:%M:%S"))
        self.locations = self.order_list['O_location'].unique()
        self.network = net.Network()
        # self.shortest_path = pd.read_csv(self.cfg.shortest_path_file)

        with open(self.cfg.OD_pickle_path, 'rb') as f:
            self.OD_dict =  pickle.load(f)
        
        with open(self.cfg.graph_pickle_file, 'rb') as f:
            self.G: nx.Graph = pickle.load(f)


        if self.cfg.optimize_target:
            # print('len OD',len(self.OD_dict), 'len Theta', len(self.cfg.Theta))
            for key in self.OD_dict.keys():
                # print('key',key)
                # print(self.OD_dict[key])

                self.OD_dict[key]['prob_shortest_path'] = self.cfg.Theta[key] 

        self.OD_results = dict()
        for OD_id, OD in self.OD_dict.items():
            self.OD_results[OD_id] = {"O_location": OD['start'], "D_location": OD['destination'],'his_order_num':0,'responded_order_num':0, 'pooled_order_num':0, 'waitingtime':[], 'detour_distance':[], 'pickup_time':[], 'shared_distance':[], 'total_travel_distance':[], 'saved_travel_distance':[]}


        self.time_unit = 10  # 控制的时间窗,每10s匹配一次
        self.index = 0  # 计数器
        self.device = cfg.device
        self.total_reward = 0
        self.optimazition_target = cfg.optimazition_target  # 仿真的优化目标
        self.matching_condition = cfg.matching_condition  # 匹配时是否有条件限制
        self.pickup_distance_threshold = cfg.pickup_distance_threshold
        self.detour_distance_threshold = cfg.detour_distance_threshold
        self.vehicle_list = []

        self.takers = []
        self.current_seekers = []  # 存储需要匹配的乘客
        self.remain_seekers = []
        self.time_reset()

        self.total_order_num = 0
        self.response_order_num = 0

        for i in range(self.vehicle_num):
            random.seed(i)
            location = random.choice(self.locations)
            vehicle = Vehicle.Vehicle(i, location, self.cfg)
            self.vehicle_list.append(vehicle)
            # 重置司机的时间
            vehicle.activate_time = self.time

        # system metric
        self.his_order = []  # all orders responsed
        self.waitingtime = []
        self.detour_distance = []
        self.traveltime = []
        self.pickup_time = []
        self.platform_income = []
        self.shared_distance = []
        self.reposition_time = []
        self.total_travel_distance = []
        self.saved_travel_distance = []


        self.matching_pairs = []

        self.carpool_order = []


    def time_reset(self):
        # 转换成时间数组
        self.time = time.strptime(
            self.cfg.date + self.cfg.simulation_begin_time, "%Y-%m-%d %H:%M:%S")
        # 转换成时间戳
        self.time = time.mktime(self.time)
        self.time_slot = 0
        # print('time reset:', self.time)

    def step(self,):
        time_old = self.time
        self.time += self.time_unit
        self.time_slot += 1

        # 筛选时间窗内的订单
        current_time_orders = self.order_list[self.order_list['beginTime_stamp'] >time_old]
        current_time_orders = current_time_orders[current_time_orders['beginTime_stamp'] <= self.time]
        self.current_seekers = []
        self.current_seekers_location = []
        for index, row in current_time_orders.iterrows():
            for key in self.OD_dict.keys():
                if self.OD_dict[key]['start'] == row['O_location'] and self.OD_dict[key]['destination'] == row['D_location']:
                    od_id = key
                    break
            seeker = Seeker.Seeker( row)
            self.OD_results[od_id]['his_order_num'] += 1
            self.total_order_num += 1

            seeker.od_id = od_id
            seeker.set_shortest_path(self.get_path(
                seeker.O_location, seeker.D_location))
            value = self.cfg.unit_distance_value / 1000 * seeker.shortest_distance
            seeker.set_value(value)
            self.current_seekers.append(seeker)
            self.current_seekers_location .append(seeker.O_location)


            np.random.seed(index)  # 设置随机种子
            if self.cfg.optimize_target:

                if np.random.rand() < self.OD_dict[key]['prob_shortest_path']:
                    seeker.path = 'shortest'
                else:
                    seeker.path = 'potential_pool'
            else:
                if np.random.rand() < self.prob:

                    seeker.path = 'shortest'
                else:
                    seeker.path = 'potential_pool'



        start = time.time()
        reward, done = self.process(self.time)
        end = time.time()
        # print('process 用时', end - start)
        return self.OD_results,  done

    #
    def process(self, time_, ):
        reward = 0
        takers = []
        vehicles = []
        seekers = self.current_seekers

        if self.time >= time.mktime(time.strptime(self.cfg.date + self.cfg.simulation_end_time, "%Y-%m-%d %H:%M:%S")):
            print('当前episode仿真时间结束,奖励为:', self.total_reward)

            # 计算系统指标
            self.res = {}
            self.res_dic = {}

            # for seekers
            for order in self.his_order:
                self.waitingtime.append(order.waiting_time)
                self.traveltime.append(order.traveltime)
            for order in self.carpool_order:
                self.detour_distance.append(order.detour)
            self.res['waitingTime'] = np.mean(self.waitingtime)
            self.res_dic['waitingTime'] = self.waitingtime

            self.res['traveltime'] = np.mean(self.traveltime)
            self.res_dic['traveltime'] = self.traveltime

            self.res['detour_distance'] = np.mean(self.detour_distance)
            self.res_dic['detour_distance'] = self.detour_distance

            # for vehicle
            self.res['pickup_time'] = np.mean(self.pickup_time)
            self.res_dic['pickup_time'] = self.pickup_time

            # self.res['shared_distance'] = np.mean(self.shared_distance)
            # self.res_dic['shared_distance'] = self.shared_distance

            self.res['total_ride_distance'] = np.sum(self.total_travel_distance)
            self.res['mean_ride_distance'] = np.mean(self.total_travel_distance)

            self.res['saved_ride_distance'] = np.sum(
                self.saved_travel_distance)
            self.res['mean_saved_ride_distance'] = np.mean(
                self.saved_travel_distance)
            
            self.res_dic['saved_ride_distance'] = self.saved_travel_distance
            

            self.res['platform_income'] = np.sum(self.platform_income)
            self.res_dic['platform_income'] = self.platform_income

            self.res['response_rate'] = len(list(set(self.his_order))) / self.total_order_num
            # print('not set',len(self.his_order) / self.total_order_num )
            self.res['carpool_rate'] = len(self.carpool_order) / len(self.his_order)



            self.save_figure('Waiting Time',self.waitingtime)
            self.save_figure('Travel Time',self.traveltime,)
            self.save_figure('Pickup Time',self.pickup_time,)
            self.save_figure('Profit',self.platform_income,)

            print('pool rate', self.res['carpool_rate'])
            print('total shared distance', np.sum(self.shared_distance))
            print('total detour distance', np.sum(self.detour_distance))
            print('Total driver occupied time',np.sum(self.pickup_time) + np.sum(self.traveltime))

            # self.save_figure('Distance Savings',self.saved_travel_distance,)
            self.save_figure('Shared Distance',self.shared_distance,)
            self.save_figure('Detour Distance',self.detour_distance,)

            # it = iter(self.carpool_order)
            # for a, b in zip(it, it):  # 每次取两个元素
            #     print(a.show(), b.show())
            #     print('**********************')

            # for i in range(len(self.matching_pairs)):
            #     print(self.matching_pairs[i][0].show(), self.matching_pairs[i][1].show(), self.matching_pairs[i][2])
            #     print('**********************')


            return reward, True
        else:
            # print('当前episode仿真时间:',time_)
            # 判断智能体是否能执行动作
            for vehicle in self.vehicle_list:
                vehicle.is_activate(time_)
                vehicle.reset_reposition()

                if vehicle.state == 1:  # 能执行动作
                    # print('id{},vehicle.target{}'.format(vehicle.id, vehicle.target))
                    if vehicle.target == 0:
                        vehicles.append(vehicle)
                    else:
                        # print('id{},vehicle.target{}'.format(vehicle.id, vehicle.target))
                        takers.append(vehicle)
            # print('len(vehicles)',len(vehicles),'len(takers)', len(takers))
            start = time.time()
            reward = self.first_protocol_matching(takers, vehicles, self.current_seekers, self.remain_seekers)
            end = time.time()
            if self.cfg.progress_target:
                print('匹配用时{},time{},vehicles{},takers{},seekers{}'.format(end - start, self.time_slot, len(vehicles),len(takers) ,len(seekers)))   
            # for vehicle in self.vehicle_list:
            #     print('vehicle.activate_time',vehicle.activate_time)
            #     print('vehicle.target',vehicle.target)
            self.total_reward += reward

            return reward,  False

    # 匹配算法，最近距离匹配
    def first_protocol_matching(self, takers, vehicles, seekers, remain_seekers  ):
        if (len(seekers) == 0 + len(remain_seekers) == 0) or len(takers) + len(vehicles) == 0:
            return 0
        import time
        start = time.time()


        # print('len(seekers)',len(seekers))
        # print('len(remain_seekers)',len(remain_seekers))

        for seeker in seekers:
            # 当前seeker的zone
            location = seeker.O_location
            zone = self.network.Nodes[location].getZone()
            nodes = zone.nodes
            pool_match = {}
            vacant_match = {}
            for taker in takers:
                if taker.location in nodes:
                    pool_match[taker] = self.calTakersWeights(seeker, taker)

            for vehicle in vehicles:
                if vehicle.location in nodes:
                    vacant_match[vehicle] = self.calVehiclesWeights(seeker, vehicle)

            # 計算匹配結果
            # 无车可用
            if not pool_match or min(pool_match.values()) == self.cfg.dead_value:

                if not vacant_match:
                    seeker.response_target = 0

                elif min(vacant_match.values()) == self.cfg.dead_value:
                    seeker.response_target = 0

                else:
                 # 派空车
                    key, weight = min(vacant_match.items(), key=lambda item: item[1])
                    # print('weight',weight,'seeker.id',seeker.id, 'vacant.id',key.id)

                    seeker.response_target = 1
                    key.order_list.append(seeker)
                    key.p0_pickup_distance = weight  # 记录seeker接驾距离
                    key.reposition_target = 0 # 重置司机的状态
                    self.pickup_time.append(self.cfg.unit_driving_time * weight)  # 记录seeker接驾时间
                    self.OD_results[seeker.od_id]['pickup_time'].append(self.cfg.unit_driving_time * weight)
                    
                    pickup_path = nx.shortest_path(self.G, key.location, key.order_list[0].O_location, weight="weight")
                    pickup_path_length =  [self.G[u][v]["weight"] for u, v in zip(pickup_path[:-1], pickup_path[1:])]
                    # 规划路径
                    if seeker.path == 'shortest':
                        key.path = pickup_path + self.OD_dict[seeker.od_id]['shortest_path'][1:]
                        key.path_length =pickup_path_length +  self.OD_dict[seeker.od_id]['shortest_path_link_length_list']
                        # print('pickup_path',pickup_path,'shotest_path',self.OD_dict[seeker.od_id]['shortest_path'])
                    else:
                        key.path =pickup_path + self.OD_dict[seeker.od_id]['high_potential_path'][1:]
                        key.path_length = pickup_path_length + self.OD_dict[seeker.od_id]['high_potential_path_length_list']
                        # print('high_potential pickup_path',pickup_path, self.OD_dict[seeker.od_id]['high_potential_path'])                    # print('self.OD_dict[seeker.od_id]',self.OD_dict[seeker.od_id])
                    # print('1 len(vehicle.path)',len(key.path), 'len(vehicle.path_length)',len(key.path_length))

                    self.OD_results[seeker.od_id]['responded_order_num'] += 1
                    self.his_order.append(seeker)

                    self.response_order_num += 1
                    
            # 先派载人车                    
            else:

                # 直接找到最大值对应的 key 和权重
                key, weight = min(pool_match.items(), key=lambda item: item[1])

                seeker.response_target = 1
                # 处理最匹配的 key
                key.order_list.append(seeker)
                key.reposition_target = 0

                # 记录等待时间
                self.waitingtime.append(self.time - seeker.begin_time_stamp)
                self.OD_results[seeker.od_id]['waitingtime'].append(self.time - seeker.begin_time_stamp)
                # print('weight',weight,'seeker.id',seeker.id, 'taker.id',key.id)
                # 移除已匹配的 taker
                takers.remove(key)
                self.pickup_time.append(weight * self.cfg.unit_driving_time)  # 记录seeker接驾距离
                self.OD_results[seeker.od_id]['pickup_time'].append(weight * self.cfg.unit_driving_time)  # 记录seeker接驾时间


                # 计算派送顺序，判断是否FIFO
                fifo, distance = self.is_fifo(key.order_list[0], key.order_list[1])

                if fifo:
                    # 先上先下
                    d1 = self.get_path(key.location, key.order_list[1].O_location)
                    d2 = self.get_path(key.order_list[1].O_location, key.order_list[0].D_location)
                    d3 = self.get_path(key.order_list[0].D_location, key.order_list[1].D_location)

                    key.drive_distance += (d1 + d2)

                    p0_invehicle = key.drive_distance
                    p0_expected_distance = key.order_list[0].shortest_distance

                    key.drive_distance += d3

                    # 绕行计算
                    key.order_list[0].set_detour(p0_invehicle - p0_expected_distance)
                    self.OD_results[key.order_list[0].od_id]['detour_distance'].append(p0_invehicle - p0_expected_distance)

                    p1_invehicle = d2 + d3
                    p1_expected_distance = key.order_list[1].shortest_distance
                    key.order_list[1].set_detour(p1_invehicle - p1_expected_distance)
                    self.OD_results[key.order_list[1].od_id]['detour_distance'].append(p1_invehicle - p1_expected_distance)

                    # Travel time
                    key.order_list[0].set_traveltime(self.cfg.unit_driving_time * p0_invehicle)
                    key.order_list[1].set_traveltime(self.cfg.unit_driving_time * p1_invehicle)

                else:
                    # 先上后下
                    d1 = self.get_path(key.location, key.order_list[1].O_location)
                    d2 = self.get_path(key.order_list[1].O_location, key.order_list[1].D_location)
                    d3 = self.get_path(key.order_list[1].D_location, key.order_list[0].D_location)

                    key.drive_distance += (d1 + d2 + d3)

                    p0_invehicle = key.drive_distance
                    p0_expected_distance = key.order_list[0].shortest_distance

                    # 绕行计算
                    key.order_list[0].set_detour(p0_invehicle - p0_expected_distance)
                    self.OD_results[key.order_list[0].od_id]['detour_distance'].append(p0_invehicle - p0_expected_distance)

                    p1_invehicle = d2
                    p1_expected_distance = key.order_list[1].shortest_distance
                    key.order_list[1].set_detour(p1_invehicle - p1_expected_distance)
                    self.OD_results[key.order_list[1].od_id]['detour_distance'].append(p1_invehicle - p1_expected_distance)

                    # Travel time
                    key.order_list[0].set_traveltime(self.cfg.unit_driving_time * p0_invehicle)
                    key.order_list[1].set_traveltime(self.cfg.unit_driving_time * p1_invehicle)

                # 计算乘客的行驶距离
                key.order_list[0].ride_distance = p0_invehicle
                self.OD_results[key.order_list[0].od_id]['total_travel_distance'].append(p0_invehicle)
                key.order_list[1].ride_distance = p1_invehicle
                self.OD_results[key.order_list[1].od_id]['total_travel_distance'].append(p1_invehicle)
                key.order_list[0].shared_distance = d2
                self.OD_results[key.order_list[0].od_id]['shared_distance'].append(d2)
                key.order_list[1].shared_distance = d2
                self.OD_results[key.order_list[1].od_id]['shared_distance'].append(d2)

                # 计算节省的行驶距离
                saved_dis = key.order_list[1].shortest_distance + key.order_list[0].shortest_distance + d1 - (p0_invehicle + p1_invehicle - d2)
                self.saved_travel_distance.append(saved_dis)
                self.OD_results[key.order_list[0].od_id]['saved_travel_distance'].append(saved_dis / 2)
                self.OD_results[key.order_list[1].od_id]['saved_travel_distance'].append(saved_dis / 2)

                # 计算平台收益
                self.platform_income.append(
                    self.cfg.discount_factor * (key.order_list[0].value + key.order_list[1].value) -
                    self.cfg.unit_distance_cost / 1000 * (d1 + d2 + d3)
                )

                # 更新智能体可以采取的动作时间
                self.total_travel_distance.append(d1 + d2 + d3)
                dispatching_time = self.cfg.unit_driving_time * (d1 + d2 + d3)
                key.activate_time = self.time + dispatching_time

                # 更新智能体的位置
                key.location = key.order_list[1].D_location
                key.origin_location = key.order_list[1].D_location

                # 完成订单
                self.his_order.append(seeker)

                self.carpool_order.append(key.order_list[0])
                self.carpool_order.append(key.order_list[1])
                
                self.matching_pairs.append((key.order_list[0], key.order_list[1],fifo ,  ))

                self.response_order_num += 1

                # 记录响应和拼车订单
                self.OD_results[seeker.od_id]['responded_order_num'] += 1
                self.OD_results[seeker.od_id]['pooled_order_num'] += 1
                self.OD_results[key.order_list[0].od_id]['pooled_order_num'] += 1

                # 重置车的状态
                key.order_list = []
                key.target = 0  # 变成vehicle
                key.drive_distance = 0
                key.reward = 0
                key.p0_pickup_distance = 0
                key.p1_pickup_distance = 0
                key.path = []
                key.path_length = []

                # 记录seeker的等待时间
                seeker.set_waitingtime(self.time - seeker.begin_time_stamp)
                seeker.response_target = 1   


        # 给剩余的seeker指派
        for seeker in remain_seekers:
            
            # 当前seeker的zone
            location = seeker.O_location
            zone = self.network.Nodes[location].getZone()
            nodes = zone.nodes
            pool_match = {}
            vacant_match = {}

            for vehicle in vehicles:
                if vehicle.location in nodes:
                    vacant_match[vehicle] = self.calVehiclesWeights(seeker, vehicle)

            # 計算匹配結果
            # 无车可用
            if not vacant_match:
                seeker.response_target = 0

            elif min(vacant_match.values()) == self.cfg.dead_value:
                seeker.response_target = 0

            else:
                # 派空车
                key, weight = min(vacant_match.items(), key=lambda item: item[1])
                # print('weight',weight,'seeker.id',seeker.id, 'vacant.id',key.id)
                seeker.response_target = 1
                key.order_list.append(seeker)
                key.p0_pickup_distance = weight  # 记录seeker接驾距离
                self.pickup_time.append(self.cfg.unit_driving_time * weight)  # 记录seeker接驾时间
                self.OD_results[seeker.od_id]['pickup_time'].append(self.cfg.unit_driving_time * weight)
                key.location = seeker.O_location
                pickup_path = nx.shortest_path(self.G, key.location, key.order_list[0].O_location, weight="weight")
                pickup_path_length =  [self.G[u][v]["weight"] for u, v in zip(pickup_path[:-1], pickup_path[1:])]
                # 规划路径
                if seeker.path == 'shortest':
                    key.path = pickup_path + self.OD_dict[seeker.od_id]['shortest_path'][1:]
                    key.path_length =pickup_path_length +  self.OD_dict[seeker.od_id]['shortest_path_link_length_list']
                    # print('pickup_path',pickup_path,'shotest_path',self.OD_dict[seeker.od_id]['shortest_path'])
                else:
                    key.path =pickup_path + self.OD_dict[seeker.od_id]['high_potential_path'][1:]
                    key.path_length = pickup_path_length + self.OD_dict[seeker.od_id]['high_potential_path_length_list']
                    # print('high_potential pickup_path',pickup_path, self.OD_dict[seeker.od_id]['high_potential_path'])
            

                key.reposition_target = 0 
                self.OD_results[seeker.od_id]['responded_order_num'] += 1

                self.his_order.append(seeker)
                self.response_order_num += 1

        end = time.time()
        # print('指派用时', end - start)
        # 更新司機位置
        for vehicle in vehicles:
            if not vehicle.order_list: # 沒接到乘客
                vehicle.reposition_target = 1
                destination = random.choice(self.locations)
                vehicle.path = nx.shortest_path(self.G, vehicle.location, destination, weight="weight")
                vehicle.path_length = [self.G[u][v]["weight"] for u, v in zip(vehicle.path[:-1], vehicle.path[1:])]
                # print('0 len(vehicle.path)',len(vehicle.path), 'len(vehicle.path_length)',len(vehicle.path_length))

        for taker in takers:
            if len(taker.order_list) != 2: # 沒接到乘客
                taker.reposition_target = 1


        start = time.time()
        # 更新位置
        for taker in takers:
            # 先判断taker 接没接到乘客
            # 当前匹配没拼到新乘客
            if taker.reposition_target == 1:
                
                # 判断是否接到第一个乘客
                if self.time - taker.order_list[0].begin_time_stamp - taker.order_list[0].waiting_time < self.cfg.unit_driving_time * taker.p0_pickup_distance:
                    continue # 还没接到
                
                # 没匹配到其他乘客，但接到p1了，开始在路上派送
                else:
                    # 没送到目的地
                    if taker.path:
                        # print('id',taker.id,'!!!!!!!',taker.path)
                        dis = self.get_path(taker.location,taker.path[0])
                        # print('taker_path',taker.path,'O',taker.location,'D',taker.path[0],'path_length',dis)
                        if self.time - taker.activate_time > self.cfg.unit_driving_time * dis:
                            taker.location = taker.path.pop(0)  # 修改为FIFO队列模式
                            
                            taker.drive_distance += dis if taker.path else 0
                            taker.activate_time = self.time
                    else:
                        # 送到目的地了
                        taker.location = taker.order_list[0].D_location
                        taker.origin_location = taker.order_list[0].D_location
                        taker.activate_time = self.time
                        
                        # 乘客的行驶距离
                        taker.order_list[0].ride_distance = taker.drive_distance
                        self.OD_results[taker.order_list[0].od_id]['total_travel_distance'].append(taker.drive_distance)

                        taker.order_list[0].shared_distance = 0
                    
                        self.total_travel_distance.append( taker.p0_pickup_distance + taker.order_list[0].ride_distance)
                        

                        # 计算平台收益
                        self.platform_income.append(taker.order_list[0].value -
                                                    (self.cfg.unit_distance_cost / 1000) * taker.drive_distance)
                        

                        taker.order_list.clear()
                        taker.target = 0  # 变成vehicle
                        taker.drive_distance = 0
                        taker.reward = 0
                        taker.p0_pickup_distance = 0
                        taker.path.clear()


            
        for vehicle in vehicles:
            if vehicle.reposition_target == 1:
                # 更新空车司机位置   
                if len(vehicle.path) > 1:
                    # 没到目的地
                    # print('vehicle.path',vehicle.path,'vehicle.reposition_time',vehicle.reposition_time,'self.time',self.time)
                    if self.time > vehicle.reposition_time + self.cfg.unit_driving_time * vehicle.path_length[0]:
                        vehicle.reposition_time = self.time
                        vehicle.location = vehicle.path.pop(0)
                        vehicle.path_length.pop(0)
                else:
                    # 到目的地了
                    # print('到目的地了',vehicle.location)
                    vehicle.reposition_time = self.time

            else:
                vehicle.target = 1  # 变成taker
                vehicle.origin_location = vehicle.location

                if len(vehicle.path) > 1:
                    # 没到目的地
                    # print('len(path)',len(vehicle.path),'len(path_length)',len(vehicle.path_length))
                    # print('vehicle.path',vehicle.path,'vehicle.reposition_time',vehicle.reposition_time,'self.time',self.time,',vehicle.path_length',vehicle.path_length)
                    if self.time > vehicle.reposition_time + self.cfg.unit_driving_time * vehicle.path_length[0]:
                        vehicle.reposition_time = self.time
                        vehicle.location = vehicle.path.pop(0)
                        vehicle.path_length.pop(0)
                else:
                    # 到目的地了
                    # print('到目的地了',vehicle.location)
                    vehicle.reposition_time = self.time


        end = time.time()
        # print('派送用时{}'.format(end-start))

        self.remain_seekers = []
        for seeker in seekers:
            if seeker.response_target == 0 and (self.time - seeker.begin_time_stamp) < self.cfg.delay_time_threshold:
                seeker.set_delay(self.time)
                self.remain_seekers.append(seeker)

        return 0



    def calTakersWeights(self,  seeker, taker):
        # 权重设计为接驾距离
        pick_up_distance = self.get_path(
            seeker.O_location, taker.location)
        
        fifo, distance = self.is_fifo(taker.order_list[0], seeker)

        if fifo:
            # 先上先下
            d1 = self.get_path(
                taker.location, seeker.O_location)
            d2 = self.get_path(
                seeker.O_location, taker.order_list[0].D_location)
            d3 = self.get_path(
                taker.order_list[0].D_location, seeker.D_location)
            

            p0_invehicle = taker.drive_distance +  (d1 + d2)
            p0_expected_distance =  taker.order_list[0].shortest_distance


            # 绕行
            p0_detour = p0_invehicle - p0_expected_distance


            p1_invehicle = d2 +d3
            p1_expected_distance = seeker.shortest_distance
            p1_detour = p1_invehicle - p1_expected_distance

        else:
            # 先上后下
            d1 = self.get_path(
                taker.location, seeker.O_location)
            d2 = self.get_path(
                seeker.O_location,seeker.D_location)
            d3 = self.get_path(
                seeker.D_location, taker.order_list[0].D_location)

            p0_invehicle = taker.drive_distance + (d1 + d2 + d3)
            p0_expected_distance = taker.order_list[0].shortest_distance


            # 绕行
            p0_detour = p0_invehicle - p0_expected_distance
                

            p1_invehicle = d2
            p1_expected_distance = seeker.shortest_distance
            p1_detour = p1_invehicle - p1_expected_distance 
            
        if pick_up_distance < self.cfg.pickup_distance_threshold and (p0_detour < max(1000,min( 0.4 *p0_expected_distance, self.detour_distance_threshold)) and
                                   p1_detour < max(1000,min( 0.4 *p1_expected_distance, self.detour_distance_threshold) )):
            return pick_up_distance
        else:
            return self.cfg.dead_value


    def calVehiclesWeights(self, seeker, vehicle ):
        # 权重设计为接驾距离
        pick_up_distance = self.get_path(
            seeker.O_location, vehicle.location)
        
        if pick_up_distance < self.cfg.pickup_distance_threshold:
            return pick_up_distance
        else:
            return self.cfg.dead_value


    def is_fifo(self, p0, p1):
        fifo = [self.get_path(p1.O_location, p0.D_location),
                self.get_path(p0.D_location, p1.D_location)]
        lifo = [self.get_path(p1.O_location, p1.D_location),
                self.get_path(p1.D_location, p0.D_location)]
        if sum(fifo) < sum(lifo):
            return True, fifo
        else:
            return False, lifo


    def get_path(self, O, D):
        # tmp = self.shortest_path[(self.shortest_path['O'] == O) & (
        #     self.shortest_path['D'] == D)]
        # if tmp['distance'].unique():
        #     return tmp['distance'].unique()[0]
        # else:
        #     return self.network.get_path(O,D)[0]
        tmp = nx.shortest_path_length(self.G, O, D, weight="weight")
        return tmp

    # def get_path_by_OD(self, OD_id):
    #     tmp = self.shortest_path_pickle[OD_id][-1]
    #     return tmp

    def save_figure(self, name, metric):
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")  # You can choose different styles based on your preference

        plt.figure(figsize=(8, 6))  # Set the figure size
        # print('metric:{}'.format(name),metric)
        # Use boxplot() functio \n
        sns.boxplot(data=metric, palette="Set3", width=0.5)

        # Customize labels and title
        plt.xlabel(name, fontsize=14)
        plt.ylabel("Value", fontsize=14)

        # Show the plot
        plt.savefig('./figure/The boxplot of {}.png'.format(name))
        plt.close()


    def save_metric(self, path):
        import pickle

        with open(path, "wb") as tf:
            pickle.dump(self.res, tf)

    def save_his_order(self, path):
        dic = {}
        for i in range(len(self.his_order)):
            dic[i] = self.his_order[i]
        import pickle

        with open(path, "wb") as tf:
            pickle.dump(dic, tf)
