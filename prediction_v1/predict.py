"""
不动点迭代过程及预测拼车概率、期望行驶里程与绕行里程
Tues Dec 7 2021
Copyright (c) 2021 Yuzhen FENG
"""
from os import error
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import time
import os
from progress.bar import Bar
from settings import params

start_time = time.time()
# -------------------- Start the fixed point iteration --------------------
# ---------- Load data ----------
with open(params['link_pickle'], 'rb') as f:
    link_dict: dict = pickle.load(f)
with open(params['OD_pickle_file'], 'rb') as f:
    OD_dict: dict = pickle.load(f)
with open(params['shortest_path_pickle'], 'rb') as f:
    path_dict: dict = pickle.load(f)
with open(params['node_pickle_file'], 'rb') as f:
    node_dict: dict = pickle.load(f)
# isPrestored = input("Do you want to use the result of the last executation? (y/n) ")
isPrestored = 'n'
if isPrestored == 'n':
    match_df = pd.read_csv(params['match_csv'], index_col=["seeker_id", "taker_id", "link_idx"])
    matches = dict()
    for index, row in match_df.iterrows():
        matches.setdefault(index[0], []).append({"taker_id": index[1], "taker_route_type": row['taker_route_type'], "taker_route_path": row['taker_route_path'],"link_idx": index[2], "preference": row["preference"],
        "ride_seeker": row["ride_seeker"], "ride_taker": row["ride_taker"],
        "detour_seeker": row["detour_seeker"], "detour_taker": row["detour_taker"], "shared": row["shared"], "destination": row["destination"],"eta_match": 0})

    print("Data loaded.")
    # ---------- Initialize seekers and takers ----------
    seekers = dict()
    TOTAL_LAMBDA = 0
    for seeker_id, OD in OD_dict.items():
        if seeker_id not in matches.keys():
            continue
        TOTAL_LAMBDA += OD['lambda']
        seekers[seeker_id] = {"lambda": OD['lambda'], "p_seeker": 0.1,'lambda_1':0,'lambda_2':0,'lambda_3':0,'lambda_4':0,'max_length':4,'mu_w':0.01 } # params['K']/len(OD_dict)

    takers = dict()

    for taker_id in OD_dict.keys():
        # 初始化 takers 字典
        takers[taker_id] = {
            'taker_origin':{0: {
                "tau_bar": params['pickup_time'],
                "lambda_taker": 0,
                "p_taker": 0.1,
                "rho_taker": 1e-2,
                "eta_taker": 1e-2
            }},
            'shortest_path':{},
            'highest_potential_path':{},
        }
        
        # 获取 taker 的所有路径
        paths = {
            'shortest_path': OD_dict[taker_id]['shortest_path'], 
            'highest_potential_path': OD_dict[taker_id]['high_potential_path']
        }

        # 遍历路径并初始化字典
        for key, path in paths.items():
            for link_idx, link_id in enumerate(path):
                takers[taker_id][key][link_idx + 1] = {
                    "tau_bar": link_dict[link_id][2] / params['speed'],
                    "lambda_taker": 0,
                    "p_taker": 0.1,
                    "rho_taker": 1e-2,
                    "eta_taker": 1e-2
                }

else:
    with open("variables/seekers.pickle", 'rb') as f:
        seekers: dict = pickle.load(f)
    with open("variables/takers.pickle", 'rb') as f:
        takers: dict = pickle.load(f)
    with open("variables/matches.pickle", 'rb') as f:
        matches: dict = pickle.load(f)
print("Variables initialized.")
# ---------- Start to iterate ----------
iter_start_time = time.time()
all_steps = []
iter_num = 0
error = params['M']

# ---------- variables to optimize ----------
def generate_random_numbers(length):
    # 生成一组正随机数
    random_numbers = np.random.rand(length)
    # 将随机数归一化，使其加和为 1
    np.random.seed(42)
    normalized_numbers = random_numbers / random_numbers.sum()
    return normalized_numbers

Pi = {}
for od_id in seekers.keys():
    Pi[od_id] = seekers[od_id]['lambda']/TOTAL_LAMBDA

print('sum(PI)',sum(Pi.values()))
PROB= params['shortest_path_prob']

Theta = {}
for OD_id in OD_dict.keys():
    Theta[OD_id] = {
        'shortest_path':PROB,
        'highest_potential_path':1 - PROB
    }

# ---------- Start to iterate ----------

print("Inner Iterating... |", end='')
# while iter_num < params['max_iter_time'] and error > params['convergent_condition'] or iter_num < params["min_iter_time"]:
while iter_num < 20: 
    print('Iteration:',iter_num, 'error',error)
    lambda_taker_step = []
    p_seeker_step = []
    p_taker_step = []
    rho_taker_step = []



    # update eta

    for seeker_id, takers_of_seeker in matches.items():
        seekers[seeker_id]["matching_pair_num"] = len(takers_of_seeker)
        eta_match_product = seekers[seeker_id]["lambda"]

        for taker in takers_of_seeker:
            taker["eta_match"] = eta_match_product
            takers[taker["taker_id"]][taker["taker_route_type"]][taker['link_idx']]["eta_taker"] += eta_match_product
            # print('taker id', taker["taker_id"], 'eta',takers[taker["taker_id"]][taker["taker_route_type"]][taker['link_idx']]["eta_taker"])
            # print('rho',takers[taker["taker_id"]][taker["taker_route_type"]][taker['link_idx']]["rho_taker"])
            eta_match_product *= 1 - takers[taker["taker_id"]][taker["taker_route_type"]][taker['link_idx']]["rho_taker"]
    
    # update p_seeker
    for seeker_id in seekers.keys():
        origin_p_seeker = seekers[seeker_id]["p_seeker"]
        product = 1
        # print('seeker_id',seeker_id) 
        # for taker in matches[seeker_id]:
        #     if taker["taker_route_type"] == 'taker_origin':
        #         product *= 1 - takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]]["rho_taker"]
        #     else:
        #         product *= 1 - takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]]["rho_taker"] * Theta[taker["taker_id"]][taker["taker_route_type"]] 
        for taker in matches[seeker_id]:
            # if takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]]["rho_taker"] >= 1e-4:
            product *= 1 - takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]]["rho_taker"]
        
        # print('product',product,'rho_taker',takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]]["rho_taker"])
        seekers[seeker_id]["p_seeker"] = 1 - product
        tolerance = 1e-9
        if abs(seekers[seeker_id]["p_seeker"] - 1) < tolerance:
            # print('seeker_id',seeker_id, 'product',product, 'len(matches)',len(matches[seeker_id]) )
            seekers[seeker_id]["p_seeker"] = seekers[seeker_id]["p_seeker"] -1e-5
        p_seeker_step.append(abs(seekers[seeker_id]["p_seeker"] - origin_p_seeker))
        # print('p_seeker',seekers[seeker_id]["p_seeker"],'origin_p_seeker',origin_p_seeker)
        # update lambda_1
        seekers[seeker_id]["lambda_1"] = seekers[seeker_id]["lambda"] * seekers[seeker_id]["p_seeker"]

        # update Pn_s(w)

        # print('lambda',seekers[seeker_id]["lambda"],'lambda_1',seekers[seeker_id]["lambda_1"])
        lambda_ = seekers[seeker_id]["lambda"] - seekers[seeker_id]["lambda_1"] 
        mu_ = seekers[seeker_id]["mu_w"] 
        # print('lambda_',lambda_,'mu',mu_,)

        def Cal_Pn_s(lambda_, mu, N, tau_w):
            if mu == 0 or tau_w <= 0:
                raise ValueError("mu must be nonzero and tau_w must be a positive integer.")
            if lambda_ <=0:
                print('fatal error')
                return 0
            rho = lambda_ / mu
            alpha = ((1+ 4 * rho)** 0.5 -1) /2
            if rho >= 1:
                # print('rho > 1')
                return 1/(tau_w + 2 )
            else:
                return alpha**N * lambda_ * (1-alpha) / (lambda_+mu*(1-alpha))

        if mu_ != 0 and lambda_ >= 0:
            
            Pn_s = [Cal_Pn_s(lambda_, mu_, i, seekers[seeker_id]["max_length"]) for i in range (0,5)]
            Pn_s_0_prime = Pn_s[0] * mu_ / lambda_ if lambda_ !=0 else 0

            # print('Pn_s_0',Pn_s_0_prime)
            # print('Pn_s',Pn_s , 'total_P',sum(Pn_s ) ,  sum(Pn_s  )+Pn_s_0_prime )

            # update lambda_3
            seekers[seeker_id]["lambda_3"] = sum(    i *  lambda_ for index, i in enumerate(Pn_s) if index >= 2 )
            # print('lambda_3',seekers[seeker_id]["lambda_3"])

            # update lambda_4
            # seekers[seeker_id]["lambda_4"] = lambda_  * ( Pn_s_0_prime + Pn_s[1] + Pn_s[0] )
            seekers[seeker_id]["lambda_4"] = lambda_  * ( Pn_s[1] + Pn_s[0] )
            # print('lambda_4',seekers[seeker_id]["lambda_4"])

            
            # # update lambda_2
            # seekers[seeker_id]["lambda_2"] = ( seekers[seeker_id]["lambda"] - seekers[seeker_id]["lambda_1"] ) * np.exp(-mu_ * seekers[seeker_id]["max_length"])

        else:
            # update lambda_3
            seekers[seeker_id]["lambda_3"] = 1e-8
            # update lambda_4
            seekers[seeker_id]["lambda_4"] = 1e-8

    # update p_taker, rho_taker 
    for OD_id, W in takers.items(): # loop for OD w
        for path_idx, taker in W.items(): # loop for path r
            if path_idx == 'taker_origin':
                continue
            for link_idx, link in taker.items(): # loop for link a

                # update Pn_t(a,r,w)
                # print('taker',taker)
                # print('path_idx',path_idx)
                # print('Theta[OD_id]',Theta[OD_id][path_idx])
                # print('taker[link_idx -1]["lambda_taker"]',taker[link_idx -1]["lambda_taker"])
                if link_idx == 1:
                    lambda_ = seekers[OD_id]["lambda_4"] * Theta[OD_id][path_idx] 
                    # print('1 lambda_',lambda_,'lambda_4',seekers[OD_id]["lambda_4"],'theta',Theta[OD_id][path_idx] )
                else:
                    # print('OD_id',OD_id,'link_idx',link_idx,'taker',link,)
                    lambda_ = taker[link_idx -1]["lambda_taker"] * (np.exp(-link["eta_taker"] * link['tau_bar']))
                    # print('OD_id',OD_id,'link_idx',link_idx,'former_lambda_taker',taker[link_idx -1]["lambda_taker"],'now_lambda_taker',lambda_, 'inner', (np.exp(-link["eta_taker"] * link['tau_bar'])) ,'eta', link["eta_taker"],'tau', link['tau_bar'] )
                    # print('2 lambda_',lambda_,)
                
                mu_ = link["eta_taker"]
                
                origin_p_taker = link["p_taker"]
                origin_rho_taker = link["rho_taker"]

                link["rho_taker"] =  1 - np.exp(- lambda_ * link["tau_bar"] )

                if mu_ == 0 :
                    link["p_taker"] = 0
                    link["rho_taker"] =  1 - np.exp(- lambda_ * link["tau_bar"] )

                else:
                    if lambda_ >= mu_:
                        # print('Pn_t',Pn_t,'total',sum(Pn_t))
                        # update p_taker, rho_taker 
                        link["p_taker"] =  mu_/lambda_
                        link["rho_taker"] =  1 - np.exp(-(lambda_-mu_)*link["tau_bar"] )

                    else: 
                        alpha = lambda_ / mu_
                        # Pn_t = [(1-alpha)* alpha**n for n in range (5)]
                        epsilon = 1e-6
                        Pn_t = [(1-alpha) * alpha**n / (1 - max(alpha ** 3, epsilon)) for n in range (10)]
                        link["p_taker"] =  lambda_ / mu_
                        link["rho_taker"] =  1 - Pn_t[0]

                # print('p_taker', link["p_taker"] , 'rho_taker',link["rho_taker"] )

                link["eta_taker"] = 0
                p_taker_step.append(abs(link["p_taker"] - origin_p_taker))
                rho_taker_step.append(abs(link["rho_taker"] - origin_rho_taker))

                # update lambda_taker
                origin_lambda_taker = takers[OD_id][path_idx][link_idx]["lambda_taker"]
                takers[OD_id][path_idx][link_idx]["lambda_taker"] = lambda_
                # print('lambda_diff', lambda_ - origin_lambda_taker, 'origin_lambda_taker',origin_lambda_taker, 'now_taker',lambda_,'taker_id',OD_id,'path_id',path_idx,'link_id',link_idx)
                # print('rho_taker error', abs(link["rho_taker"] - origin_rho_taker), 'origin_rho_taker',origin_rho_taker, 'now_rho_taker',link["rho_taker"])
                # print('p_taker error', abs(link["p_taker"] - origin_p_taker), 'origin_p_taker',origin_p_taker, 'now_p_taker',link["p_taker"])
                lambda_taker_step.append(abs(lambda_ - origin_lambda_taker))


    # update matching prob
    for seeker_id in seekers.keys():
        prob = 0
        # print("Checking seeker_id:", seeker_id)

        # if seeker_id not in Theta:
        #     print(f"seeker_id {seeker_id} not in Theta")

        # if seeker_id not in takers:
        #     print(f"seeker_id {seeker_id} not in takers")

        # if seeker_id not in seekers:
        #     print(f"seeker_id {seeker_id} not in seekers")


        for path in ['shortest_path','highest_potential_path']:
            # print('path',path,'id',seeker_id)
            # prob += Theta[seeker_id][path] * takers[seeker_id][path][len(takers[seeker_id]) - 1]["lambda_taker"] * (1 - takers[seeker_id][path][len(takers[seeker_id]) - 1]["p_taker"]) / seekers[seeker_id]["lambda_4"]
            prob += Theta[seeker_id][path] * takers[seeker_id][path][len(takers[seeker_id][path])]["lambda_taker"] * (1 - takers[seeker_id][path][len(takers[seeker_id][path]) ]["p_taker"]) / seekers[seeker_id]["lambda_4"]


        seekers[seeker_id]["matching_prob"] = (seekers[seeker_id]["lambda_1"] + seekers[seeker_id]["lambda_3"]  + (1 - prob ) * seekers[seeker_id]["lambda_4"] ) / seekers[seeker_id]["lambda"] if seekers[seeker_id]["lambda"] != 0 else 0
        # print('seeker_id',seeker_id,'taker_prob',prob,'total_prob',seekers[seeker_id]["matching_prob"] )

    # update vacant vehicle flow
    v_dic = {}
    for node_id in node_dict.keys():

        v_i = 0
        for seeker_id in seekers.keys():

            P_w = seekers[seeker_id]['matching_prob']
            v_i += (1-P_w) * seekers[seeker_id]["lambda_4"] + seekers[seeker_id]["lambda_3"]
            # print('P_w',P_w,'lambda_4',seekers[OD_index]["lambda_4"],'lambda_3',seekers[OD_index]["lambda_3"])

        for seeker_id, takers_of_seeker in matches.items():
            for taker in takers_of_seeker:
                if taker['destination'] == node_id:
                    v_i += taker["eta_match"] * takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]]["rho_taker"]
                    # print('eta',taker["eta_match"],'rho',takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]]["rho_taker"],'v_i',v_i)
        v_dic[node_id] = v_i
    

    for OD_id in OD_dict.keys():
        print('OD id',OD_id, 'mu_old',seekers[OD_id]['mu_w'])
        seekers[OD_id]['mu_w'] = 0
        for index, value in v_dic.items():
            seekers[OD_id]['mu_w'] += v_dic[index] * Pi[OD_id]
        print('OD_id',seekers[OD_id],'mu',seekers[OD_id]['mu_w'],'v_dic',v_dic[index],'Pi',Pi[index][OD_id])
    # print('v_dic',v_dic)

    
    iter_num += 1
    # if iter_num >= params["min_iter_time"]:
    #     all_steps.append([np.max(lambda_taker_step), np.max(p_seeker_step), np.max(p_taker_step), np.max(rho_taker_step)])
    #     print('error: lambda taker',np.max(lambda_taker_step), 'p_seeker', np.max(p_seeker_step), 'p_taker',np.max(p_taker_step), 'rho_taker',np.max(rho_taker_step) )
    #     print('p_seeker error', p_seeker_step)
    #     print('rho_taker error', rho_taker_step)
    #     print('p_taker error', p_taker_step)
    
    if iter_num >= params["min_iter_time"]:
        all_steps.append([
            np.max(lambda_taker_step), 
            np.max(p_seeker_step), 
            np.max(p_taker_step), 
            np.max(rho_taker_step)
        ])
        
        # def print_large_errors(name, array):
        #     large_errors = [(i, val) for i, val in enumerate(array) if abs(val) > 0.01]
        #     if large_errors:
        #         print('length:', len(large_errors), f'error in {name}:', large_errors, )
        
        # print_large_errors('lambda_taker', lambda_taker_step)
        # print_large_errors('p_seeker', p_seeker_step)
        # print_large_errors('p_taker', p_taker_step)
        # print_large_errors('rho_taker', rho_taker_step)


        error = np.max(all_steps[len(all_steps) - 1])
iter_end_time = time.time()
print("\nConverge! It costs:", iter_end_time - iter_start_time)
print("The average time of iteration:", (iter_end_time - iter_start_time) / iter_num)
# print('seeekers,',seekers)

# ---------- Plot the iteration ----------
plt.plot(np.arange(len(all_steps)) + params["min_iter_time"], all_steps, label=["lambda t(a, w)", "p s(w)", "p t(a, w)", "rho t(a, w)"])
plt.ylabel("Delta")
plt.xlabel("Iteration Time")
plt.title("Iteration")
plt.yscale("log")
plt.legend()
plt.savefig("result/iteration.png")

# ---------- Calculate the prediction result ----------

    # seekers[seeker_id]["matching_prob"] = 1 - prob 
    # print('seeker_id',seeker_id,'lambda_4',seekers[seeker_id]["lambda_4"],'prob', prob, 'numerator', (seekers[seeker_id]["lambda_1"] + seekers[seeker_id]["lambda_3"]  + (1 - prob ) * seekers[seeker_id]["lambda_4"] ), 'denominator', seekers[seeker_id]["lambda"] , 'matching_prob',seekers[seeker_id]["matching_prob"]  )
    # seekers[seeker_id]["matching_prob"] = 1 - prob + seekers[seeker_id]["p_seeker"]

for seeker_id, takers_of_seeker in matches.items():
    seekers[seeker_id]["total_ride_distance"] = 0
    seekers[seeker_id]["total_detour_distance"]= 0
    seekers[seeker_id]["total_shared_distance"]= 0
    seekers[seeker_id]["total_saved_distance"] = 0
    seekers[seeker_id]["total_matching_rate"] = 0
    # print(f"\n=== Processing Seeker ID: {seeker_id} ===")

    for taker in takers_of_seeker:
        L1 = OD_dict[seeker_id]['solo_dist']
        L2 = OD_dict[taker["taker_id"]]['solo_dist']

        seekers[seeker_id]["total_ride_distance"] += taker["ride_seeker"] * taker["eta_match"] * takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]]["rho_taker"]
        seekers[seeker_id]["total_detour_distance"] += taker["detour_seeker"] * taker["eta_match"] * takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]]["rho_taker"]
        seekers[seeker_id]["total_shared_distance"] += taker["shared"] * taker["eta_match"] * takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]]["rho_taker"]
        seekers[seeker_id]["total_saved_distance"] += (L1 + L2 - (taker["ride_seeker"] + taker["ride_taker"] - taker["shared"])) * taker["eta_match"] * takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]]["rho_taker"]
        seekers[seeker_id]["total_matching_rate"] += taker["eta_match"] * takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]]["rho_taker"]


        # print(f"Ride Distance: {seekers[seeker_id]['total_ride_distance']}, "
        #       f"Detour Distance: {seekers[seeker_id]['total_detour_distance']}, "
        #       f"Shared Distance: {seekers[seeker_id]['total_shared_distance']}, "
        #       f"Saved Distance: {seekers[seeker_id]['total_saved_distance']}, "
        #       f"Matching Rate: {seekers[seeker_id]['total_matching_rate']}")
        

        takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]]["total_ride_distance"] = takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]].setdefault("total_ride_distance", 0) + taker["ride_taker"] * taker["eta_match"] * takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]]["rho_taker"]
        takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]]["total_detour_distance"] = takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]].setdefault("total_detour_distance", 0) + taker["detour_taker"] * taker["eta_match"] * takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]]["rho_taker"]
        takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]]["total_shared_distance"] = takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]].setdefault("total_shared_distance", 0) + taker["shared"] * taker["eta_match"] * takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]]["rho_taker"]
        takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]]["total_saved_distance"] = takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]].setdefault("total_saved_distance", 0) + (L1 + L2 - (taker["ride_seeker"] + taker["ride_taker"] - taker["shared"])) * taker["eta_match"] * takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]]["rho_taker"]
        takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]]["total_matching_rate"] = takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]].setdefault("total_matching_rate", 0) + taker["eta_match"] * takers[taker["taker_id"]][taker["taker_route_type"]][taker["link_idx"]]["rho_taker"]

        # print(f"Taker ID: {taker['taker_id']}, " 
        #     f"Ride Distance: {takers[taker['taker_id']][taker['taker_route_type']][taker['link_idx']]['total_ride_distance']}, "
        #     f"Detour Distance: {takers[taker['taker_id']][taker['taker_route_type']][taker['link_idx']]['total_detour_distance']}, "
        #     f"Shared Distance: {takers[taker['taker_id']][taker['taker_route_type']][taker['link_idx']]['total_shared_distance']}, "
        #     f"Saved Distance: {takers[taker['taker_id']][taker['taker_route_type']][taker['link_idx']]['total_saved_distance']}, "
        #     f"Matching Rate: {takers[taker['taker_id']][taker['taker_route_type']][taker['link_idx']]['total_matching_rate']}")

        
        
        
for seeker_id in seekers.keys():
    L = OD_dict[seeker_id]['solo_dist']
    seekers[seeker_id]["ride_distance"] = 0
    seekers[seeker_id]["detour_distance"] = 0
    seekers[seeker_id]["shared_distance"] = 0
    seekers[seeker_id]["saved_distance"] = 0
    seekers[seeker_id]["ride_distance_for_taker"] = 0
    seekers[seeker_id]["detour_distance_for_taker"] = 0
    seekers[seeker_id]["shared_distance_for_taker"] = 0
    seekers[seeker_id]["saved_distance_for_taker"] = 0

    seekers[seeker_id]["shortest_path_prob"] = Theta[OD_id]['shortest_path'] # 选择最短路的概率

    seekers[seeker_id]["ride_distance_for_seeker"] = seekers[seeker_id]["p_seeker"] * seekers[seeker_id].setdefault("total_ride_distance", 0) / seekers[seeker_id].setdefault("total_matching_rate", params["epsilon"])
    seekers[seeker_id]["detour_distance_for_seeker"] = seekers[seeker_id]["p_seeker"] * seekers[seeker_id].setdefault("total_detour_distance", 0) / seekers[seeker_id].setdefault("total_matching_rate", params["epsilon"])
    seekers[seeker_id]["shared_distance_for_seeker"] = seekers[seeker_id]["p_seeker"] * seekers[seeker_id].setdefault("total_shared_distance", 0) / seekers[seeker_id].setdefault("total_matching_rate", params["epsilon"])
    seekers[seeker_id]["saved_distance_for_seeker"] = seekers[seeker_id]["p_seeker"] * seekers[seeker_id].setdefault("total_saved_distance", 0) / seekers[seeker_id].setdefault("total_matching_rate", params["epsilon"])

    
    seekers[seeker_id]["ride_distance"] += seekers[seeker_id]["p_seeker"] * seekers[seeker_id].setdefault("total_ride_distance", 0) / seekers[seeker_id].setdefault("total_matching_rate", params["epsilon"])
    seekers[seeker_id]["detour_distance"] += seekers[seeker_id]["p_seeker"] * seekers[seeker_id].setdefault("total_detour_distance", 0) / seekers[seeker_id].setdefault("total_matching_rate", params["epsilon"])
    seekers[seeker_id]["shared_distance"] += seekers[seeker_id]["p_seeker"] * seekers[seeker_id].setdefault("total_shared_distance", 0) / seekers[seeker_id].setdefault("total_matching_rate", params["epsilon"])
    seekers[seeker_id]["saved_distance"] += seekers[seeker_id]["p_seeker"] * seekers[seeker_id].setdefault("total_saved_distance", 0) / seekers[seeker_id].setdefault("total_matching_rate", params["epsilon"])

    # print(f"\n--- Seeker ID: {seeker_id} 3 Metrics ---")
    # print(f"Ride Distance for seeker: {seekers[seeker_id]['ride_distance_for_seeker']}, "
    #     f"Detour Distance for seeker: {seekers[seeker_id]['detour_distance_for_seeker']}, "
    #     f"Shared Distance for seeker: {seekers[seeker_id]['shared_distance_for_seeker']}, "
    #     f"Saved Distance for seeker: {seekers[seeker_id]['saved_distance_for_seeker']}, "
    #     f"Ride Distance: {seekers[seeker_id]['ride_distance']}, "
    #     f"Detour Distance: {seekers[seeker_id]['detour_distance']}, "
    #     f"Shared Distance: {seekers[seeker_id]['shared_distance']}, "
    #     f"Saved Distance: {seekers[seeker_id]['saved_distance']}, "
    #     f"Ride Distance for taker: {seekers[seeker_id]['ride_distance_for_taker']}, "
    #     f"Detour Distance for taker: {seekers[seeker_id]['detour_distance_for_taker']}, "
    #     f"Shared Distance for taker: {seekers[seeker_id]['shared_distance_for_taker']}, "
    #     f"Saved Distance for taker: {seekers[seeker_id]['saved_distance_for_taker']}, ")


    lambda_become_taker = seekers[seeker_id]["lambda_4"]
    # print('lambda_become_taker',lambda_become_taker)
    for path_idx, taker in takers[seeker_id].items(): # loop for OD w
        if path_idx == 'taker_origin':
            continue
        for link_idx, link in taker.items(): # loop for link a

            if link.setdefault("total_matching_rate", params["epsilon"]) == 0:
                continue
                
            seekers[seeker_id]["ride_distance_for_taker"] += taker[link_idx]["lambda_taker"]  * link["p_taker"] / lambda_become_taker * link.setdefault("total_ride_distance", 0) / link.setdefault("total_matching_rate", params["epsilon"])
            seekers[seeker_id]["detour_distance_for_taker"] += taker[link_idx]["lambda_taker"]  * link["p_taker"] / lambda_become_taker * link.setdefault("total_detour_distance", 0) / link.setdefault("total_matching_rate", params["epsilon"])
            seekers[seeker_id]["shared_distance_for_taker"] += taker[link_idx]["lambda_taker"]  * link["p_taker"] / lambda_become_taker * link.setdefault("total_shared_distance", 0) / link.setdefault("total_matching_rate", params["epsilon"])
            seekers[seeker_id]["saved_distance_for_taker"] += taker[link_idx]["lambda_taker"]  * link["p_taker"] / lambda_become_taker * link.setdefault("total_saved_distance", 0) / link.setdefault("total_matching_rate", params["epsilon"])

            # print('OD_id',OD_id,'path_idx',path_idx,'link_idx',link_idx,'ride_distance_for_taker',seekers[seeker_id]["ride_distance_for_taker"], 'lambda_taker', taker[link_idx]["lambda_taker"], 'p_taker', link["p_taker"] , 'lambda_become_taker', lambda_become_taker,'link.total_ride_distance', link.setdefault("total_ride_distance", 0), 'link.total_matching_rate', link.setdefault("total_matching_rate", params["epsilon"]))


            seekers[seeker_id]["ride_distance"] += taker[link_idx]["lambda_taker"]  * link["p_taker"] / seekers[seeker_id]["lambda"] * link.setdefault("total_ride_distance", 0) / link.setdefault("total_matching_rate", params["epsilon"])
            seekers[seeker_id]["detour_distance"] += taker[link_idx]["lambda_taker"]  * link["p_taker"] / seekers[seeker_id]["lambda"] * link.setdefault("total_detour_distance", 0) / link.setdefault("total_matching_rate", params["epsilon"])
            seekers[seeker_id]["shared_distance"] += taker[link_idx]["lambda_taker"]  * link["p_taker"] / seekers[seeker_id]["lambda"] * link.setdefault("total_shared_distance", 0) / link.setdefault("total_matching_rate", params["epsilon"])
            seekers[seeker_id]["saved_distance"] += taker[link_idx]["lambda_taker"]  * link["p_taker"] / seekers[seeker_id]["lambda"] * link.setdefault("total_saved_distance", 0) / link.setdefault("total_matching_rate", params["epsilon"])
    
    seekers[seeker_id]["ride_distance"] += (1 - seekers[seeker_id]["matching_prob"] + seekers[seeker_id]["lambda_3"] / seekers[seeker_id]["lambda"] ) * L
    seekers[seeker_id]["ride_distance_for_seeker"] += (1 - seekers[seeker_id]["p_seeker"] + seekers[seeker_id]["lambda_3"] / seekers[seeker_id]["lambda"]  ) * L

    for path in ['shortest_path','highest_potential_path']:
        seekers[seeker_id]["ride_distance_for_taker"] += takers[seeker_id][path][len(takers[seeker_id][path])]["lambda_taker"] * (1 - takers[seeker_id][path][len(takers[seeker_id][path])]["p_taker"]) / lambda_become_taker * L
        

# update matching prob
for seeker_id in seekers.keys():
    prob = 0
    # print('seeker_id',seeker_id)

    for path in ['shortest_path','highest_potential_path']:
        prob += Theta[seeker_id][path] * takers[seeker_id][path][len(takers[seeker_id][path])]["lambda_taker"] * (1 - takers[seeker_id][path][len(takers[seeker_id][path])]["p_taker"]) / seekers[seeker_id]["lambda_4"]
        # print('path',path, 'theta',Theta[seeker_id][path],'last lambda taker',takers[seeker_id][path][len(takers[seeker_id]) - 1]["lambda_taker"],  'last p_taker',takers[seeker_id][path][len(takers[seeker_id]) - 1]["p_taker"],
        #       'lambda_4', seekers[seeker_id]["lambda_4"], 'prob',prob )
    seekers[seeker_id]["matching_prob"] = (seekers[seeker_id]["lambda_1"] + seekers[seeker_id]["lambda_3"]  + (1 - prob ) * seekers[seeker_id]["lambda_4"] ) / seekers[seeker_id]["lambda"] if seekers[seeker_id]["lambda"] != 0 else 0
    # print('lambda_1',seekers[seeker_id]["lambda_1"] ,'lambda_3',seekers[seeker_id]["lambda_3"],'lambda',seekers[seeker_id]["lambda"] ,
    #         'lambda_4',seekers[seeker_id]["lambda_4"], 'matching prob', seekers[seeker_id]["matching_prob"], 'p_seeker',seekers[seeker_id]["p_seeker"], 'prob', prob
            # , 'p_taker', (seekers[seeker_id]["lambda_3"]  + (1 - prob ) * seekers[seeker_id]["lambda_4"] ) / seekers[seeker_id]["lambda"])
    
    seekers[seeker_id]["cancel_rate"] = 1 - (seekers[seeker_id]["lambda_1"] + seekers[seeker_id]["lambda_3"]  + seekers[seeker_id]["lambda_4"] ) / seekers[seeker_id]["lambda"]
    seekers[seeker_id]["rate_1"] = seekers[seeker_id]["lambda_1"] / seekers[seeker_id]["lambda"] 
    seekers[seeker_id]["rate_3"] = seekers[seeker_id]["lambda_3"] / seekers[seeker_id]["lambda"] 
    seekers[seeker_id]["rate_4"] = seekers[seeker_id]["lambda_4"] / seekers[seeker_id]["lambda"] 


# ---------- Save the prediction result to csv ----------
print("Result saving ...")
result = pd.DataFrame.from_dict(seekers, orient='index').loc[:, [
    "matching_prob", "ride_distance", "detour_distance", "shared_distance", "saved_distance",
    "ride_distance_for_taker", "detour_distance_for_taker", "shared_distance_for_taker", "saved_distance_for_taker",
    "ride_distance_for_seeker", "detour_distance_for_seeker", "shared_distance_for_seeker", "saved_distance_for_seeker",'shortest_path_prob','cancel_rate','matching_pair_num','p_seeker','rate_1','rate_3','rate_4']]
result.index.name = "OD_id"
result.to_csv(params['prediction_results_file'])

# ---------- Dump to pickle ----------
# f = open('variables/seekers.pickle', 'wb')
# pickle.dump(seekers, f)
# f.close()
# f = open('variables/takers.pickle', 'wb')
# pickle.dump(takers, f)
# f.close()
# f = open('variables/matches.pickle', 'wb')
# pickle.dump(matches, f)
# f.close()

# -------------------- End --------------------
end_time = time.time()
# ---------- Log ----------

total_distance_saving = 0
for od_id  in OD_dict.keys():
    total_distance_saving += seekers[od_id]['saved_distance'] * seekers[od_id]['lambda']

print('total_distance_saving',total_distance_saving)



