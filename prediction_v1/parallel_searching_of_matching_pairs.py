"""
匹配对的并行检索
Sun Apr 17 2022
Copyright (c) 2022 Yuzhen FENG
"""
import itertools
import os
import pickle
import networkx as nx
import pandas as pd
from multiprocessing import Pool
import time
import traceback
from settings import params

from tqdm import tqdm
from datetime import datetime


if True:
    # ---------- Load data ----------
    with open("tmp/link.pickle", 'rb') as f:
        link_dict: dict = pickle.load(f)
    with open(params['OD_pickle_file'], 'rb') as f:
        OD_dict: dict = pickle.load(f)
    with open(params['shortest_path_pickle'], 'rb') as f:
        path_dict: dict = pickle.load(f)
    # if len(OD_dict) < params['OD_num']:
    #     print("WARNING: The number of OD in OD.pickle is less than that set in settings.py")
        # quit()
    with open(params['graph_pickle_file'], 'rb') as f:
        G: nx.Graph = pickle.load(f)
    with open(params['ego_graph_pickle'], 'rb') as f:
        subgraph = pickle.load(f)


print('*************data loaded******************')


def search_matching_pairs(OD_idx1, OD_idx2, OD1, OD2, L1, L2, distance_between_dest, distance_from_seeker_origin_to_taker_dest):
    """Search matching pairs whose seeker-states belong to OD1 and taker-states to OD2

    Args:
        OD_idx1 (int): The ID of OD1 seeker
        OD_idx2 (int): The ID of OD2 taker
        OD1 (list): The information of OD1
        OD2 (list): The information of OD2
        L1 (double): The distance between OD1
        L2 (double): The distance between OD2
        distance_between_dest (double): The distance between the destinations of the two OD
        distance_from_seeker_origin_to_taker_dest (double): The distance from OD1's origin to OD2's destination

    Returns:
        list: List of matching pairs
    """
    match = []
    # ---------- get the subgraph of neighbors centered at the origin of the seeker
    nearest_G: nx.Graph = subgraph[OD_idx1]
    
    if nearest_G.has_node(OD2['start']): # 附近有给定OD2的起点 在taker起点就匹配上了
        pickup_distance = nearest_G.nodes[OD2['start']]["weight"]
        if pickup_distance <= params["search_radius"]:
            if distance_between_dest is None: # 两个终点间的距离
                try:
                    distance_between_dest = nx.shortest_path_length(G, OD1['destination'], OD2['destination'], weight="weight")
                    # distance_between_dest = shortest_paths_and_distances
                except:
                    distance_between_dest = params['M']
            if distance_from_seeker_origin_to_taker_dest is None:  # seeker起点到taker终点（先送taker，后送seeker）
                try:
                    distance_from_seeker_origin_to_taker_dest = nx.shortest_path_length(G, OD2['destination'], OD1['start'], weight="weight")
                except:
                    distance_from_seeker_origin_to_taker_dest = params['M']
            L_taker_FOLO = pickup_distance + L1 + distance_between_dest
            L_seeker_FOLO = L1
            L_taker_FOFO = pickup_distance + distance_from_seeker_origin_to_taker_dest
            L_seeker_FOFO = distance_from_seeker_origin_to_taker_dest + distance_between_dest
            detour = min(max(L_seeker_FOFO - L1, L_taker_FOFO - L2), max(L_seeker_FOLO - L1, L_taker_FOLO - L2))
            if detour < params['max_detour']: # 满足匹配条件，开始计算其他指标
                if max(L_seeker_FOFO - L1, L_taker_FOFO - L2) < max(L_seeker_FOLO - L1, L_taker_FOLO - L2):
                    ride_seeker = L_seeker_FOFO
                    ride_taker = L_taker_FOFO
                    shared = distance_from_seeker_origin_to_taker_dest
                    destination = OD2['destination']
                else:
                    ride_seeker = L_seeker_FOLO
                    ride_taker = L_taker_FOLO
                    shared = L1
                    destination = OD1['destination']
                detour_seeker = ride_seeker - L1
                detour_taker = ride_taker - L2
                # 存储指标，seeker_id, taker_id, taker_route_type, taker_route_path, matching_link_index, ride_seeker, ride_taker, detour_seeker, detour_taker, shared,destination
                prefer = params['w_detour'] * detour + params['w_pickup'] * pickup_distance + params['w_shared'] * shared + params["w_ride"] * (ride_seeker + ride_taker - shared)
                match.append([OD_idx1, OD_idx2, 'taker_origin',[-1],0, prefer, ride_seeker, ride_taker, detour_seeker, detour_taker, shared, destination])



    path_dic = {'shortest_path':OD_dict[OD_idx2]['shortest_path'],'highest_potential_path':OD_dict[OD_idx2]['high_potential_path']}
    
    for route_type ,path in path_dic.items(): # 4,最短路，5,最大拼成路径
    
        for edge in nearest_G.edges(data=True): # 在taker的行驶过程中匹配上
            # OD1附近的路网中，有OD2的path 经过的路径
            # edge 是一个三元组，包含了边的两个节点和与边关联的数据
            # edge[0] 是边的起点，edge[1] 是边的终点，edge[2] 是与边关联的数据
            link_id = edge[2]["key"]
            if link_id in path: # OD2的path经过的边
                L_taker_init = 0
                i = 0
                for path_link_id in path: # 第几段路经过OD1附近
                    i += 1
                    if path_link_id == link_id:
                        break
                    L_taker_init += link_dict[path_link_id][2] # 经过OD的路段长度
                L_taker_init += link_dict[link_id][2] / 2 # 附近路段的长度的一半
                pickup_distance = (nearest_G.nodes[link_dict[link_id][0]]["weight"] + nearest_G.nodes[link_dict[link_id][1]]["weight"]) / 2
                if pickup_distance > params["search_radius"]:
                    continue
                if distance_between_dest is None:
                    try:
                        distance_between_dest = nx.shortest_path_length(G, OD1['destination'], OD2['destination'], weight="weight")
                    except:
                        distance_between_dest = params['M']
                if distance_from_seeker_origin_to_taker_dest is None:
                    try:
                        distance_from_seeker_origin_to_taker_dest = nx.shortest_path_length(G, OD2['destination'], OD1['start'], weight="weight")
                    except:
                        distance_from_seeker_origin_to_taker_dest = params['M']
                L_taker_FOLO = L_taker_init + pickup_distance + L1 + distance_between_dest
                L_seeker_FOLO = L1
                L_taker_FOFO = L_taker_init + pickup_distance + distance_from_seeker_origin_to_taker_dest
                L_seeker_FOFO = distance_from_seeker_origin_to_taker_dest + distance_between_dest
                detour = min(max(L_seeker_FOFO - L1, L_taker_FOFO - L2), max(L_seeker_FOLO - L1, L_taker_FOLO - L2))
                if detour < params['max_detour']: # 可以匹配，计算其他指标
                    if max(L_seeker_FOFO - L1, L_taker_FOFO - L2) < max(L_seeker_FOLO - L1, L_taker_FOLO - L2):
                        if L_taker_init + pickup_distance + distance_between_dest >= L2 + L1:
                            continue    
                        ride_seeker = L_seeker_FOFO
                        ride_taker = L_taker_FOFO
                        shared = distance_from_seeker_origin_to_taker_dest
                        destination = OD2['destination'] # taker 先下车
                    else:
                        if L_taker_init + pickup_distance + distance_between_dest >= L1 + distance_from_seeker_origin_to_taker_dest:
                            continue
                        ride_seeker = L_seeker_FOLO
                        ride_taker = L_taker_FOLO
                        shared = L1
                        destination = OD1['destination'] # seeker 先下车
                    detour_seeker = max(ride_seeker - L1, 0)
                    detour_taker = max(ride_taker - L2, 0)
                    prefer = params['w_detour'] * detour + params['w_pickup'] * pickup_distance + params['w_shared'] * shared + params["w_ride"] * (ride_seeker + ride_taker - shared)
                    # 存储指标，seeker_id, taker_id, taker_route_type, taker_route_path, matching_link_index, ride_seeker, ride_taker, detour_seeker, detour_taker, shared,destination
                    match.append([OD_idx1, OD_idx2, route_type, path, i, prefer, ride_seeker, ride_taker, detour_seeker, detour_taker, shared,destination])
    return match

def generate_matches(OD_infor):
    """Search matching pairs whose taker-states belong to the OD in the parameter

    Args:
        OD_infor (tuple): The first element is the ID of the OD. The second is a list of the information of the OD including its origin node ID, destination node ID and the lambda (the mean demand)

    Raises:
        ValueError: The shortest path of the OD has not been include in shortest_path.pickle

    Returns:
        list: List of matching pairs, which is a list of [seeker's ID, taker's ID, taker's link's index, prederence, ride distance of the seeker, ride distance of the taker, detour distance of the seeker, detour distance of the taker, shared distance of the matching pair]
    """
    
    OD2_id, OD2 = OD_infor
    match_tmp = []
    # if OD2_id not in path_dict:
    #     raise ValueError("Path not found.")
    distance_from_dest = nx.single_source_dijkstra_path_length(G, source=OD2['destination'], weight="weight")
    for OD1_id, OD1 in OD_dict.items():
        L1 = path_dict[OD1_id][-1]
        L2 = path_dict[OD2_id][-1]
        # L1_ = shortest_paths_and_distances[(shortest_paths_and_distances['source'] == OD1[0]) & (shortest_paths_and_distances['target'] == OD1[1]) ]['distance'].iloc[0]
        # L2_ = shortest_paths_and_distances[(shortest_paths_and_distances['source'] == OD1[0]) & (shortest_paths_and_distances['target'] == OD1[1]) ]['distance'].iloc[0]
        # print('L1:{},L1_dif:{}'.format(L1,L1_),'L2:{},L2_dif:{}'.format(L2,L2_))
        try:
            distance_between_dest = distance_from_dest[OD1['destination']]
            distance_from_seeker_origin_to_taker_dest = distance_from_dest[OD1['start']]
        except KeyError:
            distance_between_dest = distance_from_seeker_origin_to_taker_dest = None
            
        try:
            match_ = search_matching_pairs(OD1_id, OD2_id, OD1, OD2, L1, L2, distance_between_dest, distance_from_seeker_origin_to_taker_dest)
            match_tmp += match_
        except KeyError:
                traceback.print_exc()
    return match_tmp

        
if __name__ == '__main__':
    all_start_time = time.time()
    param = {'process_num': 32, 'chunk_num': 5}

    with Pool(processes=param['process_num']) as pool:
        print("Start searching matching pairs: ")
        start_time = time.time()

        tasks = pool.imap_unordered(generate_matches, OD_dict.items())
        results = list(tqdm(tasks, total=len(OD_dict), desc="Processing"))

        end_time = time.time()
        print("Finish searching matching pairs:", end_time - start_time)

    # 合并结果
    result = list(itertools.chain.from_iterable(results))
    m = pd.DataFrame(result, columns=["seeker_id", "taker_id","taker_route_type", 
                                      "taker_route_path",  "link_idx", "preference", 
                                      "ride_seeker", "ride_taker", "detour_seeker", 
                                      "detour_taker", "shared", 'destination'])

    # 重新排序
    m = m.sort_values(by=["seeker_id", "preference"], ascending=[True, False])

    # 保存结果
    m.to_csv(params['match_csv'], index=False)
    print("There are", m.shape[0], "matching pairs in all.")


        