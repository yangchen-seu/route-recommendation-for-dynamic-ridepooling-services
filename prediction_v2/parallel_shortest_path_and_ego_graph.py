"""
最短路径的并行检索
Sun Apr 17 2022
Copyright (c) 2022 Yuzhen FENG
"""
import os
import pickle
import networkx as nx
from heapq import heappush, heappop
from itertools import count
from multiprocessing import Pool
import time
from settings import params

if True:
    # ---------- Load data ----------
    with open("tmp/OD.pickle", 'rb') as f:
        OD_dict: dict = pickle.load(f)
    # if len(OD_dict) < params['OD_num']:
    #     print("WARNING: The number of OD in OD.pickle is less than that set in settings.py")
    #     quit()
    with open("tmp/graph.pickle", 'rb') as f:
        G: nx.Graph = pickle.load(f)

def new_dijkstra_multisource(
    G, sources, weight, pred=None, paths=None, cutoff=None, target=None
):
    G_succ = G._succ if G.is_directed() else G._adj

    push = heappush
    pop = heappop
    dist = {}  # dictionary of final distances
    seen = {}
    # fringe is heapq with 3-tuples (distance,c,node)
    # use the count c to avoid comparing nodes (may not be able to)
    c = count()
    fringe = []
    for source in sources:
        seen[source] = 0
        push(fringe, (0, next(c), source))
    while fringe:
        (d, _, v) = pop(fringe)
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        if v == target:
            break
        for u, e in G_succ[v].items():
            cost = weight(v, u, e)
            if cost is None:
                continue
            vu_dist = dist[v] + cost
            if cutoff is not None:
                if dist[v] + cost / 2 > cutoff:
                    continue
            if u in dist:
                u_dist = dist[u]
                if vu_dist < u_dist:
                    raise ValueError("Contradictory paths found:", "negative weights?")
                elif pred is not None and vu_dist == u_dist:
                    pred[u].append(v)
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                push(fringe, (vu_dist, next(c), u))
                if paths is not None:
                    paths[u] = paths[v] + [u]
                if pred is not None:
                    pred[u] = [v]
            elif vu_dist == seen[u]:
                if pred is not None:
                    pred[u].append(v)

    # The optional predecessor and path dictionaries can be accessed
    # by the caller via the pred and paths objects passed as arguments.
    return dist

def new_weight_function(G, weight):
    if callable(weight):
        return weight
    # If the weight keyword argument is not callable, we assume it is a
    # string representing the edge attribute containing the weight of
    # the edge.
    if G.is_multigraph():
        return lambda u, v, d: min(attr.get(weight, 1) for attr in d.values())
    return lambda u, v, data: data.get(weight, 1)

def new_multi_source_dijkstra(G, sources, target=None, cutoff=None, weight="weight"):
    if not sources:
        raise ValueError("sources must not be empty")
    for s in sources:
        if s not in G:
            raise nx.NodeNotFound(f"Node {s} not found in graph")
    if target in sources:
        return (0, [target])
    weight = new_weight_function(G, weight)
    paths = {source: [source] for source in sources}  # dictionary of paths
    dist = new_dijkstra_multisource(
        G, sources, weight, paths=paths, cutoff=cutoff, target=target
    )
    if target is None:
        return (dist, paths)
    try:
        return (dist[target], paths[target])
    except KeyError as err:
        raise nx.NetworkXNoPath(f"No path to {target}.") from err

nx.algorithms.multi_source_dijkstra = new_multi_source_dijkstra # override the multi_source_dijkstra

def new_ego_graph(G, n, radius=1, center=True, undirected=False, distance=None):
    if undirected:
        if distance is not None:
            sp, _ = nx.single_source_dijkstra(
                G.to_undirected(), n, cutoff=radius, weight=distance
            )
        else:
            sp = dict(
                nx.single_source_shortest_path_length(
                    G.to_undirected(), n, cutoff=radius
                )
            )
    else:
        if distance is not None:
            sp, _ = nx.single_source_dijkstra(G, n, cutoff=radius, weight=distance)
        else:
            sp = dict(nx.single_source_shortest_path_length(G, n, cutoff=radius))

    H = G.subgraph(sp).copy()
    if not center:
        H.remove_node(n)
    return sp, H

nx.generators.ego_graph = new_ego_graph # override the ego_graph

def calculate_dijkstra_path(OD_infor):
    """Calculate the Dijkstra path for an OD

    Args:
        OD_infor (tuple): The first element is the ID of the OD. The second is a list of the information of the OD including its origin node ID, destination node ID and the lambda (the mean demand)

    Returns:
        tuple: the first element is the ID of the OD and the second is the list of edges in the shortest path from the origin to the destination
    """
    OD_id, OD = OD_infor
    (length, path) = nx.single_source_dijkstra(G, source=OD[0], target=OD[1], weight='weight')
    node_list = path
    result = list()
    for node_idx in range(len(node_list) - 1):
        result.append(G.get_edge_data(node_list[node_idx], node_list[node_idx + 1])["key"])
    result.append(length)
    # print('>', end='')
    return OD_id, result

def generate_ego_graph(OD_infor):
    """Generate the ego graph for an OD

    Args:
        OD_infor (tuple): The first element is the ID of the OD. The second is a list of the information of the OD including its origin node ID, destination node ID and the lambda (the mean demand)

    Returns:
        tuple: the first element is the ID of the OD and the second is the ego graph centered at the origin node of the OD (networkX.Graph)
    """
    OD_id, OD = OD_infor
    distance, ego_graph = nx.generators.ego_graph(G, OD[0], radius=params['search_radius'], distance="weight")
    nx.set_node_attributes(ego_graph, distance, "weight")
    # print('>', end='')
    return OD_id, ego_graph

if __name__ == '__main__':
    all_start_time = time.time()
    pool = Pool(processes=params['process_num'])
    # ---------- calculate the shortest paths ----------
    print("Start calculating shortest paths: ")
    start_time = time.time()
    result = pool.map(calculate_dijkstra_path, OD_dict.items(), len(OD_dict) // (params["process_num"] * 5))
    end_time = time.time()
    print("Finish calculating shortest paths:", end_time - start_time)
    # ---------- dump the shortest paths ----------
    f = open("tmp/shortest_path.pickle", 'wb')
    pickle.dump(dict(result), f)
    f.close()
    # ---------- generate the ego graph ----------
    print("Start generating ego graphs: ")
    start_time = time.time()
    result = pool.map(generate_ego_graph, OD_dict.items(), len(OD_dict) // (params["process_num"] * params["chunk_num"]))
    end_time = time.time()
    print("Finish generating ego graphs:", end_time - start_time)
    # ---------- dump the ego graphs ----------
    f = open("tmp/ego_graph.pickle", 'wb')
    pickle.dump(dict(result), f)
    f.close()
    # ---------- close the process pool ----------
    pool.close()
    pool.join()

    all_end_time = time.time()

    with open("log.txt", "a") as f:
        f.write(time.ctime() + ": Run " + os.path.basename(__file__) + " with Params = " + str(params) + "; Cost " + str(all_end_time - all_start_time) + 's\n')