
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
        print('total orders loaded:', len(self.order_list))
        self.vehicle_num = int(len(self.order_list) / cfg.order_driver_ratio )
        print('Total number of vehicles:', self.vehicle_num)
        self.order_list['beginTime_stamp'] = self.order_list['dwv_order_make_haikou_1.departure_time'].apply(
            lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
        self.begin_time = time.mktime(time.strptime(
            cfg.date + cfg.simulation_begin_time, "%Y-%m-%d %H:%M:%S"))
        self.end_time = time.mktime(time.strptime(
            cfg.date + cfg.simulation_end_time, "%Y-%m-%d %H:%M:%S"))
        self.network = net.Network()


        self.locations = self.order_list['O_location'].unique() 
        

        # self.shortest_path = pd.read_csv(self.cfg.shortest_path_file)

        with open(self.cfg.OD_pickle_path, 'rb') as f:
            self.OD_dict =  pickle.load(f)
        
        with open(self.cfg.graph_pickle_file, 'rb') as f:
            self.G: nx.Graph = pickle.load(f)



        if self.cfg.optimize_target:
            print('len OD',len(self.OD_dict), 'len Theta', len(self.cfg.Theta))
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
            # 重置为空车
            vehicle.state = 0

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
        fully_occupied = []
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



            # print('pool rate', self.res['carpool_rate'])
            # print('total shared distance', np.sum(self.shared_distance))
            # print('total detour distance', np.sum(self.detour_distance))
            # print('Total driver occupied time',np.sum(self.pickup_time) + np.sum(self.traveltime))



            return reward, True
        else:

            # 判断智能体是否能执行动作
            for vehicle in self.vehicle_list:


                if vehicle.state == 0:  # 空车
                    vehicles.append(vehicle)
                elif vehicle.state == 1:  # 有订单
                    takers.append(vehicle)
                else:
                    fully_occupied.append(vehicle)  # 已满载

            if len(vehicles) == 0 and len(takers) == 0 and len(self.current_seekers) == 0 and len(self.remain_seekers) == 0:
                print('没有订单了，当前episode仿真时间:', time_)
                return reward, True
            start = time.time()
            reward = self.first_protocol_matching(takers, vehicles, self.current_seekers, self.remain_seekers)
            end = time.time()
            self.vehicle_update(vehicles, takers, fully_occupied)  # 更新车辆状态

            end2 = time.time()
            if self.cfg.progress_target:
                print('匹配用时{},更新用时:{}, time_slot{},vehicles{},takers{},seekers{},fully_occupied:{}'.format(end - start, end2-end, self.time_slot, len(vehicles),len(takers) ,len(seekers),len(fully_occupied)))

            self.total_reward += reward

            return reward,  False

    # 匹配算法，最近距离匹配
    def first_protocol_matching(self, takers, vehicles, seekers, remain_seekers  ):

        import time
        start = time.time()



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
            if not pool_match or max(pool_match.values()) == - self.cfg.dead_value:

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
                    key.reposition_target = False # 重置司机的状态
                    key.state = 1 # 转为载人车

                    self.pickup_time.append(self.cfg.unit_driving_time * weight)  # 记录seeker接驾时间
                    self.OD_results[seeker.od_id]['pickup_time'].append(self.cfg.unit_driving_time * weight)
                    
                    pickup_path = nx.shortest_path(self.G, key.location, key.order_list[0].O_location, weight="weight")
                    pickup_path_length =  [self.G[u][v]["weight"] for u, v in zip(pickup_path[:-1], pickup_path[1:])]
                    # 规划路径
                    key.pickup_path = pickup_path
                    key.pickup_path_length = pickup_path_length

                    if seeker.path == 'shortest':

                        key.path = self.OD_dict[seeker.od_id]['shortest_path']
                        key.path_length =self.OD_dict[seeker.od_id]['shortest_path_link_length_list']
                    else:

                        key.path = self.OD_dict[seeker.od_id]['high_potential_path']
                        key.path_length = self.OD_dict[seeker.od_id]['high_potential_path_length_list']

                    self.OD_results[seeker.od_id]['responded_order_num'] += 1
                    self.his_order.append(seeker)

                    self.response_order_num += 1
                    vehicles.remove(key)  # 移除已匹配的空车

                    
            # 先派载人车                    
            else:

                # 直接找到最大值对应的 key 和权重
                key, weight = max(pool_match.items(), key=lambda item: item[1])

                seeker.response_target = 1
                # 处理最匹配的 key
                key.order_list.append(seeker)
                key.reposition_target = False
                key.state = 2 # transform into a fully occupied vehicle

                # 记录等待时间
                self.waitingtime.append(self.time - seeker.begin_time_stamp)
                self.OD_results[seeker.od_id]['waitingtime'].append(self.time - seeker.begin_time_stamp)
                # print('weight',weight,'seeker.id',seeker.id, 'taker.id',key.id)
                # 移除已匹配的 taker
                takers.remove(key)
                self.pickup_time.append(weight * self.cfg.unit_driving_time)  # 记录seeker接驾距离
                self.OD_results[seeker.od_id]['pickup_time'].append(weight * self.cfg.unit_driving_time)  # 记录seeker接驾时间

                if key.pickup_path:
                    # 如果已经有接驾路径了，则继续
                    
                    pickup_path = nx.shortest_path(self.G, key.order_list[0].O_location, key.order_list[1].O_location, weight="weight")
                    pickup_path_length =  [self.G[u][v]["weight"] for u, v in zip(pickup_path[:-1], pickup_path[1:])]

                    # 规划路径
                    key.pickup_path = key.pickup_path + pickup_path[1:]  # 连接上新的接驾路径
                    key.pickup_path_length = key.pickup_path_length + pickup_path_length


                else:

                    pickup_path = nx.shortest_path(self.G, key.location, key.order_list[0].O_location, weight="weight")
                    pickup_path_length =  [self.G[u][v]["weight"] for u, v in zip(pickup_path[:-1], pickup_path[1:])]
                    # 规划路径
                    key.pickup_path = pickup_path
                    key.pickup_path_length = pickup_path_length

                                # 完成订单
                
                self.his_order.append(seeker)

                self.carpool_order.append(key.order_list[0])
                self.carpool_order.append(key.order_list[1])
                
                # self.matching_pairs.append((key.order_list[0], key.order_list[1] ))

                self.response_order_num += 1

                # 记录响应和拼车订单
                self.OD_results[key.order_list[1].od_id]['responded_order_num'] += 1
                self.OD_results[key.order_list[1].od_id]['pooled_order_num'] += 1
                self.OD_results[key.order_list[0].od_id]['pooled_order_num'] += 1

                # 记录seeker的等待时间
                key.order_list[1].set_waitingtime(self.time - key.order_list[1].begin_time_stamp)
                key.order_list[1].response_target = 1   


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
                key.reposition_target = False # 重置司机的状态
                key.state = 1 # 转为载人车

                self.pickup_time.append(self.cfg.unit_driving_time * weight)  # 记录seeker接驾时间
                self.OD_results[seeker.od_id]['pickup_time'].append(self.cfg.unit_driving_time * weight)
                
                pickup_path = nx.shortest_path(self.G, key.location, key.order_list[0].O_location, weight="weight")
                pickup_path_length =  [self.G[u][v]["weight"] for u, v in zip(pickup_path[:-1], pickup_path[1:])]
                # 规划路径
                key.pickup_path = pickup_path
                key.pickup_path_length = pickup_path_length

                if seeker.path == 'shortest':
                    # print('shortest',seeker.id)
                    key.path = self.OD_dict[seeker.od_id]['shortest_path']
                    key.path_length =self.OD_dict[seeker.od_id]['shortest_path_link_length_list']
                else:
                    # print('high_potential',seeker.id)
                    key.path = self.OD_dict[seeker.od_id]['high_potential_path']
                    key.path_length = self.OD_dict[seeker.od_id]['high_potential_path_length_list']
                # print('self.OD_dict[seeker.od_id]',self.OD_dict[seeker.od_id])
                # print('1 len(vehicle.path)',len(key.path), 'len(vehicle.path_length)',len(key.path_length))

                self.OD_results[seeker.od_id]['responded_order_num'] += 1
                self.his_order.append(seeker)

                self.response_order_num += 1
                vehicles.remove(key)  # 移除已匹配的空车

        end = time.time()
        # print('指派用时', end - start)

        self.remain_seekers = []
        for seeker in seekers:
            if seeker.response_target == 0 and (self.time - seeker.begin_time_stamp) < self.cfg.delay_time_threshold:
                seeker.set_delay(self.time)
                self.remain_seekers.append(seeker)

        return 0


    def vehicle_update(self, vacants, takers, fully_occupied):
        # 更新空车
        for vacant in vacants:
            if len(vacant.path) > 1:
                # 没到目的地
                if self.time > vacant.reposition_time + self.cfg.unit_driving_time * vacant.path_length[0]:
                    vacant.reposition_time = self.time
                    vacant.location = vacant.path.pop(0)
                    vacant.path_length.pop(0)
                    vacant.reposition_target = False
            else:
                # 到目的地了
                vacant.reposition_time = self.time
                vacant.reposition_target = True  # 重置为需要调度
                if self.current_seekers_location:
                    # 随机选择一个新的目的地
                    vacant.destination = random.choice(self.current_seekers_location)
                else:
                    # 重置目的地
                    vacant.destination = random.choice(self.locations)  
                vacant.path = nx.shortest_path(self.G, vacant.location, vacant.destination, weight="weight")
                vacant.path_length =  [self.G[u][v]["weight"] for u, v in zip(vacant.path[:-1], vacant.path[1:])]
                

        # 更新载人车
        for taker in takers:
            # 判断是否接到乘客
            if len(taker.pickup_path) > 1: # 还没接到乘客
                if self.time > taker.activate_time + self.cfg.unit_driving_time * taker.pickup_path_length[0]:
                    taker.activate_time = self.time
                    taker.location = taker.pickup_path.pop(0)
                    taker.pickup_path_length.pop(0)

            else:  # 开始派送
                if len(taker.path) > 1:
                    # print('id',taker.id,'!!!!!!!',taker.path)

                    # print('taker_path',taker.path,'O',taker.location,'D',taker.path[0],'path_length',dis)
                    if self.time - taker.activate_time > self.cfg.unit_driving_time * taker.path_length[0]:
                        taker.location = taker.path.pop(0)  # 修改为FIFO队列模式
                        
                        taker.drive_distance += taker.path_length[0] if taker.path else 0
                        
                        taker.path_length.pop(0)  # 移除已经行驶的路径长度
                        taker.activate_time = self.time
                else:
                    # 送到目的地了
                    
                    # 乘客的行驶距离
                    taker.order_list[0].ride_distance = taker.drive_distance


                    self.OD_results[taker.order_list[0].od_id]['total_travel_distance'].append(taker.drive_distance)

                    taker.order_list[0].shared_distance = 0
                
                    self.total_travel_distance.append( taker.p0_pickup_distance + taker.order_list[0].ride_distance)
                    

                    # 计算平台收益
                    self.platform_income.append(taker.order_list[0].value -
                                                (self.cfg.unit_distance_cost / 1000) * taker.drive_distance)
                    
                    taker.reset(activate_time = self.time, location = taker.order_list[0].D_location)  # 重置taker状态  

        # 更新满载车
        for fully in fully_occupied:
            # 判断是否接到乘客
            if len(fully.pickup_path) > 1: # 还没接到乘客
               
                if self.time > fully.activate_time + self.cfg.unit_driving_time * fully.pickup_path_length[0]:
                    fully.activate_time = self.time
                    fully.location = fully.pickup_path.pop(0)
                    fully.pickup_path_length.pop(0)

            else:  # 开始派送
                # 计算派送顺序，判断是否FIFO
                fifo, distance = self.is_fifo(fully.order_list[0], fully.order_list[1])

                if fifo:
                    # 先上先下
                    d1 = self.get_path(fully.location, fully.order_list[1].O_location)
                    d2 = self.get_path(fully.order_list[1].O_location, fully.order_list[0].D_location)
                    d3 = self.get_path(fully.order_list[0].D_location, fully.order_list[1].D_location)

                    fully.drive_distance += (d1 + d2)

                    p0_invehicle = fully.drive_distance
                    p0_expected_distance = fully.order_list[0].shortest_distance

                    fully.drive_distance += d3

                    # 绕行计算
                    fully.order_list[0].set_detour(p0_invehicle - p0_expected_distance)
                    self.OD_results[fully.order_list[0].od_id]['detour_distance'].append(p0_invehicle - p0_expected_distance)

                    p1_invehicle = d2 + d3
                    p1_expected_distance = fully.order_list[1].shortest_distance
                    fully.order_list[1].set_detour(p1_invehicle - p1_expected_distance)
                    self.OD_results[fully.order_list[1].od_id]['detour_distance'].append(p1_invehicle - p1_expected_distance)

                    # Travel time
                    fully.order_list[0].set_traveltime(self.cfg.unit_driving_time * p0_invehicle)
                    fully.order_list[1].set_traveltime(self.cfg.unit_driving_time * p1_invehicle)

                    desination = fully.order_list[1].D_location

                else:
                    # 先上后下
                    d1 = self.get_path(fully.location, fully.order_list[1].O_location)
                    d2 = self.get_path(fully.order_list[1].O_location, fully.order_list[1].D_location)
                    d3 = self.get_path(fully.order_list[1].D_location, fully.order_list[0].D_location)

                    fully.drive_distance += (d1 + d2 + d3)

                    p0_invehicle = fully.drive_distance
                    p0_expected_distance = fully.order_list[0].shortest_distance

                    # 绕行计算
                    fully.order_list[0].set_detour(p0_invehicle - p0_expected_distance)
                    self.OD_results[fully.order_list[0].od_id]['detour_distance'].append(p0_invehicle - p0_expected_distance)

                    p1_invehicle = d2
                    p1_expected_distance = fully.order_list[1].shortest_distance
                    fully.order_list[1].set_detour(p1_invehicle - p1_expected_distance)
                    self.OD_results[fully.order_list[1].od_id]['detour_distance'].append(p1_invehicle - p1_expected_distance)

                    # Travel time
                    fully.order_list[0].set_traveltime(self.cfg.unit_driving_time * p0_invehicle)
                    fully.order_list[1].set_traveltime(self.cfg.unit_driving_time * p1_invehicle)

                    desination = fully.order_list[0].D_location

                # 计算乘客的行驶距离
                fully.order_list[0].ride_distance = p0_invehicle
                self.OD_results[fully.order_list[0].od_id]['total_travel_distance'].append(p0_invehicle)
                fully.order_list[1].ride_distance = p1_invehicle
                self.OD_results[fully.order_list[1].od_id]['total_travel_distance'].append(p1_invehicle)
                fully.order_list[0].shared_distance = d2
                self.OD_results[fully.order_list[0].od_id]['shared_distance'].append(d2)
                fully.order_list[1].shared_distance = d2
                self.OD_results[fully.order_list[1].od_id]['shared_distance'].append(d2)

                # 计算节省的行驶距离
                saved_dis = fully.order_list[1].shortest_distance + fully.order_list[0].shortest_distance + d1 - (p0_invehicle + p1_invehicle - d2)
                self.saved_travel_distance.append(saved_dis)
                self.OD_results[fully.order_list[0].od_id]['saved_travel_distance'].append(saved_dis / 2)
                self.OD_results[fully.order_list[1].od_id]['saved_travel_distance'].append(saved_dis / 2)

                # 计算平台收益
                self.platform_income.append(
                    self.cfg.discount_factor * (fully.order_list[0].value + fully.order_list[1].value) -
                    self.cfg.unit_distance_cost / 1000 * (d1 + d2 + d3)
                )

                # 更新智能体可以采取的动作时间
                self.total_travel_distance.append(d1 + d2 + d3)
                dispatching_time = self.cfg.unit_driving_time * (d1 + d2 + d3)
                
                fully.reset(activate_time = self.time + dispatching_time, location = desination)  # 重置taker状态





    def calTakersWeights(self,  seeker, taker):
        # 权重设计为接驾距离
        if taker.pickup_path:
            # 还没接到上一个乘客
            pick_up_distance = self.get_path(
            seeker.O_location, taker.order_list[0].O_location) 
        else:
            # 已经接到了
            pick_up_distance = self.get_path(seeker.O_location, taker.location)
        
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
            

        saved_dis = seeker.shortest_distance + taker.order_list[0].shortest_distance + d1 - (p0_invehicle + p1_invehicle - d2)

        if pick_up_distance < self.cfg.pickup_distance_threshold and (p0_detour <  self.detour_distance_threshold) and   ( p1_detour < self.detour_distance_threshold) :
            
            return saved_dis
        
            return pick_up_distance
        else:
            return - self.cfg.dead_value


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

        tmp = nx.shortest_path_length(self.G, O, D, weight="weight")
        return tmp

    # def get_path_by_OD(self, OD_id):
    #     tmp = self.shortest_path_pickle[OD_id][-1]
    #     return tmp


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
