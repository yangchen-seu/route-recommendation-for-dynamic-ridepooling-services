"""
加载数据并缓存为二进制文件
Sun Nov 28 2021
Copyright (c) 2021 Jiayu MA, Yuzhen FENG
"""
import pandas as pd
import pickle
import networkx as nx
from progress.bar import Bar
import time
import os
import settings

start_time = time.time()

# ------------------------------ Start to load data ------------------------------
# ---------- Load and dump node.csv ----------
bar = Bar("Loading node.csv", fill='#', max=100, suffix='%(percent)d%%')
node_data = pd.read_csv("./data/node.csv")  # node文件
node_num = node_data.shape[0]  # 节点数
node_dict = dict()

for i in bar.iter(range(node_num)):
    node_dict[int(node_data.loc[i].loc["node_id"])] = [node_data.loc[i].loc["x_coord"],
                                                       node_data.loc[i].loc["y_coord"],
                                                       node_data.loc[i].loc["W_node_id"],]  # 列表[经度，纬度,终点为该点的ODindex]

f = open('tmp/node.pickle', 'wb')
pickle.dump(node_dict, f)
f.close()
# ---------- Load and dump link.csv ----------
bar = Bar("Loading link.csv", fill='#', max=100, suffix='%(percent)d%%')
link_data = pd.read_csv("./data/links.csv")  # link文件
link_num = link_data.shape[0]  # 边数
link_dict = dict()

for i in bar.iter(range(link_num)):
    link_dict[int(link_data.index[i])] = [
    int(link_data.loc[i].loc["from_node_id"]),  # 起点 ID
    int(link_data.loc[i].loc["to_node_id"]),    # 终点 ID
    link_data.loc[i].loc["length"]              # 边的长度
]
print('link_data',len(link_data), 'link_num',link_num, 'link_dict',len(link_dict))

f = open('tmp/link.pickle', 'wb')
pickle.dump(link_dict, f)
f.close()

# ---------- Build a graph ----------
G = nx.Graph()
G.add_nodes_from(node_dict.keys())
for link_id, link in link_dict.items():
    if not G.has_edge(link[0], link[1]):
        G.add_edge(link[0], link[1], weight=link[2], key=link_id)
print(len(G.edges))

f = open('tmp/graph.pickle', 'wb')
pickle.dump(G, f)
f.close()





# ---------- Load and dump OD.csv ----------
bar = Bar("Loading OD.csv", fill='#', max=100, suffix='%(percent)d%%')
OD_data = pd.read_csv( settings.params['OD_file'])  # OD文件
OD_num = OD_data.shape[0]  # OD数
OD_dict = dict()
print('OD_num',OD_num)

with open("tmp/graph.pickle", 'rb') as f:
    G: nx.Graph = pickle.load(f)
        
import networkx as nx


# ---------- Load and dump PATH.csv ----------
with open('tmp/shortest_distances.pickle', 'rb') as f:
    SHORTEST_DISTANCES = pickle.load(f)
    
def find_max_weight_path_dag_with_length(G, source, target):
    # 获取拓扑排序
    topological_order = list(nx.topological_sort(G))
    
    # 初始化最大权路径字典，起点权值为0，其他为负无穷
    max_weight = {node: float('-inf') for node in G.nodes}
    max_weight[source] = 0
    
    # 辅助字典存储前驱节点以便回溯路径
    predecessor = {node: None for node in G.nodes}
    
    # 遍历拓扑排序后的节点
    for u in topological_order:
        if max_weight[u] == float('-inf'):
            continue  # 如果节点不可达，跳过
        for v, attr in G[u].items():
            weight = attr['passenger_number']  # 当前用于优化的权值
            if max_weight[u] + weight > max_weight[v]:
                max_weight[v] = max_weight[u] + weight
                predecessor[v] = u
    
    # 如果目标节点不可达
    if max_weight[target] == float('-inf'):
        return None, float('-inf'), 0  # 不可达时返回路径、权值和权重长度为0
    
    # 回溯路径并计算路径长度（以 weight 为权重）
    path = []
    total_weight_length = 0  # 路径的权重长度
    current = target
    while current is not None:
        if predecessor[current] is not None:  # 累加边的权重
            total_weight_length += G[predecessor[current]][current]['weight']
        path.append(current)
        current = predecessor[current]
    path.reverse()
    
    return path, max_weight[target], total_weight_length

def remove_nodes_after_a(G, a):
    # 确保图是有向无环图
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("The graph is not a DAG.")

    # 获取拓扑排序
    topo_sort = list(nx.topological_sort(G))

    # 找到节点 a 的位置
    if a not in topo_sort:
        raise ValueError(f"Node {a} is not in the graph.")

    a_index = topo_sort.index(a)

    # 获取节点 a 之后的所有节点
    nodes_to_remove = topo_sort[a_index + 1:]

    # 从图中移除这些节点
    G.remove_nodes_from(nodes_to_remove)

    return G



def construct_subgraph(G, os, od_star, alpha = 1.5):
    """
    构建子图 D = (V_D, E_D)
    :param G: 完整的路网图 (NetworkX DiGraph)
    :param os: 起点 (订单的 pickup 点)
    :param od_star: 目标点 (订单的 drop-off 点)
    :return: 子图 D (NetworkX DiGraph)
    """

    sp_os_to_od_star = SHORTEST_DISTANCES[os][od_star]
    
    # 确定子图节点集 V_D
    V_D = {
        v for v in G.nodes
        if SHORTEST_DISTANCES[os][v] + SHORTEST_DISTANCES[v][od_star] <= alpha * sp_os_to_od_star
    }
    
    # 确定子图边集 E_D
    E_D = [
        (u, v) for u, v in G.edges
        if u in V_D and v in V_D and
            SHORTEST_DISTANCES[v][od_star]<=
           SHORTEST_DISTANCES[u][od_star]
    ]
    for u, v in G.edges:
        if u in V_D and v in V_D:
            if SHORTEST_DISTANCES[u][od_star] <= SHORTEST_DISTANCES[v][od_star]:
               E_D.append((v,u))
    
    # 构建子图
    D = nx.DiGraph()
    D.add_nodes_from(V_D)
    # 添加边和边属性
    for u, v in E_D:
        if G.has_edge(u, v):
            attributes = G.get_edge_data(u, v)  # 获取原图中的边属性
            D.add_edge(u, v, **attributes)
    
    # 计算拓扑排序
    topological_order = list(nx.topological_sort(D))

    # 输出节点及其拓扑排序
    topological_order_dict = {node: i+1 for i, node in enumerate(topological_order)}  # 排序从1开始

    # 如果需要将拓扑顺序作为属性存储到图中
    nx.set_node_attributes(D, topological_order_dict, name="topological_order")


    return D


passenger_number = pd.read_csv(settings.params['OD_passenger_num_file'])
passenger_number.head()
tmp = passenger_number.copy()


  
for i in bar.iter(range(OD_num)):
    star = int(OD_data.loc[i].loc["O_location"])
    dest = int(OD_data.loc[i].loc["D_location"])

    (shortest_length, shortest_path) = nx.single_source_dijkstra(G, source=star, target=dest, weight='weight')

    D = construct_subgraph(G, star, dest)
    D_ = remove_nodes_after_a(D, dest)
    nx.set_edge_attributes(D_, 0, name="passenger_number")

    tmp.loc[:, 'O_location'] = tmp['O_location'].astype(int)


    for node in D_.nodes:
        # 检查当前节点是否在 tmp['O_location'] 中
        if node in tmp['O_location'].values:
            # 找到对应的行，获取对应的 OrderCount
            row = tmp[tmp['O_location'] == node].iloc[0]
            OrderCount = row['OrderCount']
            
            # 遍历图中所有从 node 出发的边，设置 passenger_number 为 OrderCount
            for u, v in D_.out_edges(node):  # 使用 out_edges 获取出发边
                D_[u][v]['passenger_number'] = OrderCount  # 设置边的 passenger_number 属性
            
    path, max_passengers, path_weight_length = find_max_weight_path_dag_with_length(D_, star, dest)
    
    OD_dict[int(OD_data.loc[i].loc["OD_id"])] = [star,
                                                 dest,
                                                 OD_data.loc[i].loc["lambda"],
                                                 shortest_length, # OD_data.loc[i].loc["solo_distance"],
                                                 shortest_path,
                                                 path,
                                                 max_passengers, 
                                                 path_weight_length,
                                                 1]  # 列表[起点id，终点id，lambda, solo_dist,shortest_path, highest_path,max_passenger, max_passenger_length]

    
    
f = open('tmp/OD.pickle', 'wb')
pickle.dump(OD_dict, f)
f.close()




# ------------------------------ End to load data ------------------------------
end_time = time.time()
# ---------- Log ----------
with open("log.txt", "a") as f:
    f.write(time.ctime() + ": Run " + os.path.basename(__file__) + " with Params = " + str(settings.params) + "; Cost " + str(end_time - start_time) + 's\n')
