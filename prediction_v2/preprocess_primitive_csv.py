"""
处理原始路网，简化道路，包括：选取主干道、检查连通性、合并二度节点
Wed Dec  1 2021
Copyright (c) 2021 Yuzhen FENG
"""
import pandas as pd
import numpy as np
import networkx as nx
from settings import params

# ---------- Load primitive csv files ----------
node = pd.read_csv("primitive/node.csv", index_col="node_id", encoding="ISO-8859-1")
link = pd.read_csv("primitive/link.csv", index_col="link_id", encoding="ISO-8859-1")

# ---------- Select links and nodes ----------
selected_link = link.loc[link.loc[:, "link_type"] <= params["lowest_road_class"], :]
from_node = selected_link.loc[:, "from_node_id"]
to_node = selected_link.loc[:, "to_node_id"]
selected_node_idx = pd.concat([from_node, to_node]).drop_duplicates()
selected_node = node.loc[selected_node_idx, :]
selected_link = link.loc[np.all(np.c_[link["from_node_id"].isin(selected_node_idx), link["to_node_id"].isin(selected_node_idx)], axis=1), :]

# Check connectivity
G: nx.Graph = nx.from_pandas_edgelist(selected_link, "from_node_id", "to_node_id", ["length", "geometry", "link_type"])
print("Graph with " + str(G.number_of_nodes()) + " nodes and " + str(G.number_of_edges()) + " links.")
print("Select maximum subgraph from connected component with " + str([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]) + " nodes.")
connected_G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
# Combine two degree link
def combine_geometry(g1, g2):
    g1_str: str = g1[12:-1]
    g2_str: str = g2[12:-1]
    origin_1 = g1_str.split(', ')[0]
    dest_1 = g1_str.split(', ')[-1]
    origin_2 = g2_str.split(', ')[0]
    dest_2 = g2_str.split(', ')[-1]
    if origin_1 == dest_2:
        return 'LINESTRING (' + g2_str + ', ' + g1_str + ')'
    else:
        return 'LINESTRING (' + g1_str + ', ' + g2_str + ')'
for mid in list(connected_G.nodes):
    neighbors = list(connected_G.neighbors(mid))
    if len(neighbors) == 2:
        u = neighbors[0]; v = neighbors[1]
        length_u = connected_G.get_edge_data(u, mid)["length"]
        length_v = connected_G.get_edge_data(v, mid)["length"]
        if length_u + length_v > params["max_combined_length"]:
            continue
        geometry_u: str = connected_G.get_edge_data(u, mid)["geometry"]
        link_type_u = connected_G.get_edge_data(u, mid)["link_type"]
        geometry_v = connected_G.get_edge_data(v, mid)["geometry"]
        link_type_v = connected_G.get_edge_data(v, mid)["link_type"]
        connected_G.add_edge(u, v, length=length_u + length_v, geometry=combine_geometry(geometry_u, geometry_v), link_type=np.random.choice([link_type_u, link_type_v]))
        connected_G.remove_node(mid)
print("Simplified graph with " + str(connected_G.number_of_nodes()) + " nodes and " + str(connected_G.number_of_edges()) + " links.")
print("Graph connectivity: " + str(nx.is_connected(connected_G)))

selected_link = nx.to_pandas_edgelist(connected_G)
selected_link = selected_link.rename({"source": "from_node_id", "target": "to_node_id"}, axis=1)
selected_node = node.loc[list(connected_G.nodes), :]

# ---------- Generate new indices and save old indices of all nodes ----------
selected_node = selected_node.reset_index()
selected_node.index.name = "index"
selected_node = selected_node.set_index("node_id", append=True)
selected_node = selected_node.reset_index()
selected_node = selected_node.set_index("node_id")

# ---------- Update node indices of all links ----------
from_node_new_id: pd.Series = selected_node.loc[selected_link.loc[:, "from_node_id"], "index"]
from_node_new_id = from_node_new_id.reset_index(drop=True)
to_node_new_id: pd.Series = selected_node.loc[selected_link.loc[:, "to_node_id"], "index"]
to_node_new_id = to_node_new_id.reset_index(drop=True)
selected_link = selected_link.reset_index(drop=True)
selected_link.loc[:, "from_node_id"] = from_node_new_id
selected_link.loc[:, "to_node_id"] = to_node_new_id

# ---------- Save it ----------
selected_node = selected_node.reset_index(drop=True)
selected_link = selected_link.reset_index(drop=True)
selected_node.index.name = "node_id"
selected_link.index.name = "link_id"
selected_node.iloc[:, 1:].to_csv("./data/node.csv", encoding='utf-8')
selected_link.to_csv("./data/link.csv", encoding='utf-8')
