'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-07-03 09:04:17
LastEditors: yangchen-seu 58383528+yangchen-seu@users.noreply.github.com
LastEditTime: 2023-01-10 01:16:55
FilePath: /matching/reinforcement learning/Seeker.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import time
import math

class Seeker():

    def __init__(self, row ) -> None:

        self.id = row['dwv_order_make_haikou_1.order_id']
        self.begin_time = row['dwv_order_make_haikou_1.departure_time']
        self.O_location = row.O_location
        self.D_location = row.D_location
        self.begin_time_stamp = time.mktime(time.strptime\
            (self.begin_time, "%Y-%m-%d %H:%M:%S"))

        self.service_target = 0
        self.detour = 0
        self.shortest_distance = 0
        self.traveltime = 0

        self.delay = 1
        self.response_target = 0
        self.k = 1
        self.value = 0
        self.shared_distance = 0
        self.ride_distance = 0

        self.carpool_target = 0 # 是否会选择拼车
        self.waiting_time = 0
        self.matching_pair = 0

    def show(self):
        print('id', self.id, 'begin_time',self.begin_time, 'O_location', self.O_location, 'D_location', self.D_location, 'detour', self.detour, 'matching_pair', self.matching_pair)

    def set_delay(self, time):
        self.k = math.floor((time - self.begin_time_stamp) / 60 )
        self.delay = 1.1 ** self.k

    def set_value(self,value):
        self.value = value
        
    def set_shortest_path(self,distance):
        self.shortest_distance = distance

    def set_waitingtime(self, waitingtime):
        self.waiting_time = waitingtime

    def set_traveltime(self,traveltime):
        self.traveltime = traveltime

    def set_detour(self,detour):
        self.detour = detour

    def cal_expected_ride_distance_for_wait(self, gamma):
        self.shared_distance  =self.shared_distance * gamma
