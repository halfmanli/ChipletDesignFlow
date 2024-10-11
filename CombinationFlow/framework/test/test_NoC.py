import argparse
import networkx as nx
from tqdm import tqdm, trange
import numpy as np
from ..NoC import *
import os
from collections import defaultdict
import json

dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
dir_tgff = os.path.join(dir_root, "tool/tgff")
dir_chaco = os.path.join(dir_root, "tool/chaco")
dir_booksim = os.path.join(dir_root, "tool/booksim")
dir_dsent = os.path.join(dir_root, "tool/dsent")


def test_tgff():
    cfg = {"task_cnt": (50, 5), "seed": np.random.randint(1e9), "tg_cnt": 50}
    gs = gen_task_tgff(dir_tgff=dir_tgff, bw=128, cfg=cfg, clean=True)
    for g in gs:
        show_graph(g, edge_attr="comm")
        print(len(g.nodes()), len(g.edges()))


def test_mincut_chaco1():
    cfg = {"task_cnt": (50, 5), "seed": np.random.randint(1e9), "tg_cnt": 500}
    gs = gen_task_tgff(dir_tgff=dir_tgff, bw=128, cfg=cfg, clean=True)
    for idx_g, g in enumerate(gs):
        for k in range(1, len(g)):
            partition = mincut_chaco(dir_chaco=dir_chaco, G=g, k=k, clean=True)
            assert len(partition) == k
            assert set([p for pt in partition for p in pt]) == set(range(len(g)))
            assert sum(map(lambda e: len(e), partition)) == len(g)
        print("graph {} PASS".format(idx_g))


def test_mincut_chaco2():
    G = nx.DiGraph()
    G.add_edge(0, 1, comm=100)
    G.add_edge(1, 2, comm=100)
    G.add_edge(2, 1, comm=200)
    G.add_edge(2, 3, comm=100)
    G.add_edge(3, 2, comm=100)
    G.add_edge(1, 4, comm=100)
    G.add_edge(2, 5, comm=100)
    G.add_edge(5, 2, comm=100)
    k = 3
    partition = mincut_chaco(dir_chaco=dir_chaco, G=G, k=k)
    assert (len(partition) == k)
    total_comm = 0
    for i in range(len(partition)):
        for j in range(i + 1, len(partition)):
            for n_i in partition[i]:
                for n_j in partition[j]:
                    total_comm += G.edges[(n_i, n_j)]["comm"] if G.has_edge(n_i, n_j) else 0
                    total_comm += G.edges[(n_j, n_i)]["comm"] if G.has_edge(n_j, n_i) else 0

    print(partition, total_comm)


def test_TPS():
    test_times = 500
    for _ in trange(test_times):
        while True:
            g = nx.gnp_random_graph(60, 0.4)
            if nx.is_connected(g):
                break

        TPS = get_TPS(g)
        g = g.to_directed(as_view=False)  # this is important
        # because acyclic if for directed graph

        nnum = len(g)
        src_edges = []
        for s in range(nnum):
            src_edges.append((nnum + s, s))
        g.add_edges_from(src_edges)

        dst_edges = []
        for d in range(nnum):
            dst_edges.append((d, nnum * 2 + d))
        g.add_edges_from(dst_edges)

        L = nx.line_graph(g)
        for (a, b, c) in TPS:
            if L.has_edge((a, b), (b, c)):
                L.remove_edge((a, b), (b, c))
            if L.has_edge((c, b), (b, a)):
                L.remove_edge((c, b), (b, a))

        assert nx.is_directed_acyclic_graph(L)
        for s in src_edges:
            for d in dst_edges:
                assert nx.has_path(L, s, d)


def test_routing_base1():
    cfg = {"task_cnt": (60, 5), "seed": np.random.randint(1e9), "tg_cnt": 300}
    rnum = 30  # router num
    gs = gen_task_tgff(dir_tgff=dir_tgff, bw=128, cfg=cfg, clean=True)
    for task_graph in tqdm(gs):
        task_size = len(task_graph)
        while True:
            topo_graph = nx.gnp_random_graph(rnum, 0.1)
            if nx.is_connected(topo_graph):
                node_mapping = dict(
                    zip(topo_graph.nodes(),
                        map(lambda n: task_size + n, sorted(topo_graph.nodes(), key=lambda k: np.random.random()))))
                topo_graph = nx.relabel_nodes(topo_graph, node_mapping)
                for c in range(task_size):
                    topo_graph.add_edge(c, task_size + np.random.randint(1e5) % rnum)  # connect to a router randomly
                topo_graph = topo_graph.to_directed(as_view=True)
                break

        TPS = get_TPS(topo_graph.to_undirected(as_view=True))
        topo_graph = topo_graph.to_directed(as_view=True)
        rtp = routing_base(task_graph=task_graph, topo_graph=topo_graph, TPS=TPS, bw=128, weight_func=None)
        assert check_deadlock(rtp)


def test_routing_base2():
    cfg = {"task_cnt": (60, 5), "seed": np.random.randint(1e9), "tg_cnt": 300}
    rnum = 30  # router num
    gs = gen_task_tgff(dir_tgff=dir_tgff, bw=128, cfg=cfg, clean=True)
    for task_graph in tqdm(gs):
        task_size = len(task_graph)
        while True:
            topo_graph = nx.gnp_random_graph(rnum, 0.1)
            if nx.is_connected(topo_graph):
                node_mapping = dict(
                    zip(topo_graph.nodes(),
                        map(lambda n: task_size + n, sorted(topo_graph.nodes(), key=lambda k: np.random.random()))))
                topo_graph = nx.relabel_nodes(topo_graph, node_mapping)
                for c in range(task_size):
                    topo_graph.add_edge(c, task_size + np.random.randint(1e5) % rnum)  # connect to a router randomly
                topo_graph = topo_graph.to_directed(as_view=True)
                break

        TPS = []
        topo_graph = topo_graph.to_directed(as_view=True)
        rtp = routing_base(task_graph=task_graph,
                           topo_graph=topo_graph,
                           TPS=TPS,
                           bw=128,
                           weight_func=lambda acc_comm, cur_comm: 1)

        # have set weight to 1 and TPS to empty, so the routing paths found should be shortest path
        for (src, dst) in task_graph.edges():
            path_routing = len(rtp[(src, dst)]) + 2
            path_shortest = nx.shortest_path(topo_graph, source=src, target=dst)
            assert path_routing == len(path_shortest)


def test_booksim():
    cfg = {"task_cnt": (60, 5), "seed": 3, "tg_cnt": 500}
    gs = gen_task_tgff(dir_tgff=dir_tgff, bw=128, cfg=cfg, clean=True)
    for task_graph in tqdm(gs):
        task_size = len(task_graph)
        topo_size = np.random.randint(20, len(task_graph))
        while True:
            topo_graph = nx.gnp_random_graph(topo_size, 0.1)
            if nx.is_connected(topo_graph):
                node_mapping = dict(
                    zip(topo_graph.nodes(),
                        map(lambda n: task_size + n, sorted(topo_graph.nodes(), key=lambda k: np.random.random()))))
                topo_graph = nx.relabel_nodes(topo_graph, node_mapping)
                for c in range(task_size):
                    topo_graph.add_edge(c, task_size + np.random.randint(1e5) % topo_size)
                topo_graph = topo_graph.to_directed()
                break

        TPS = get_TPS(topo_graph.to_undirected(as_view=True))
        rtp = routing_base(task_graph=task_graph, topo_graph=topo_graph, TPS=TPS, bw=128)

        # remove the links not used in rtp
        unused_edges = []
        for e in topo_graph.edges():
            used = False
            for sd, path in rtp.items():
                s, d = sd
                if e == (s, path[0]) or e == (path[-1], d):
                    used = True
                    break

                for e_p in zip(path, path[1:]):
                    if e == e_p:
                        used = True
                        break
                if used:
                    break
            if not used:
                unused_edges.append(e)
        topo_graph.remove_edges_from(unused_edges)

        while True:
            iso_node = list(nx.isolates(topo_graph))
            if not len(iso_node):
                break
            assert iso_node[0] >= len(task_graph)
            topo_size = len(topo_graph)
            iso_node = iso_node[0]
            topo_graph.remove_node(iso_node)
            node_mapping = dict(zip(range(iso_node + 1, topo_size), range(iso_node, topo_size - 1)))
            topo_graph = nx.relabel_nodes(topo_graph, node_mapping)

            for path in rtp.values():  # path node also need to -1
                for i in range(len(path)):
                    assert path[i] != iso_node
                    if path[i] > iso_node:
                        path[i] -= 1

            for path in rtp.values():
                for e_p in zip(path, path[1:]):
                    assert e_p in topo_graph.edges()

        # show_graph(topo_graph)

        assert check_deadlock(rtp)
        PPA = eval_PPA_booksim(dir_booksim=dir_booksim,
                               task_graph=task_graph,
                               topo_graph=topo_graph,
                               rtp=rtp,
                               cfg={
                                   "sim_cycle": 10000,
                                   "num_vcs": 4,
                                   "vc_buf_size": 4
                               },
                               clean=True)
        print(PPA, len(topo_graph))


def test_check_deadlock():
    rtp = {(0, 2): [4, 5, 6], (1, 3): [5, 6, 7], (2, 0): [6, 7, 4]}
    assert check_deadlock(rtp)
    rtp[(3, 1)] = [7, 4, 5]
    assert not check_deadlock(rtp)


def test_router_dsent():
    PA = eval_router_dsent(dir_dsent=dir_dsent,
                           in_port=5,
                           out_port=5,
                           load=0.5,
                           cfg={
                               "process": 45,
                               "num_vc": 4,
                               "vc_buf_size": 4
                           })
    print(PA)


def test_wire_dsent():
    P = eval_wire_dsent(dir_dsent=dir_dsent, load=0.5, cfg={"delay": 1e-10, "process": 45}, clean=False)
    print(P)


def test_eval_pa_router_setup():
    eval_pa_router_setup(dir_dsent=dir_dsent,
                         port_range=range(1, 9),
                         load_step=0.001,
                         processes=10,
                         cfg_router={
                             "process": 45,
                             "freq": 1e9,
                             "channel_width": 128,
                             "num_vc": 4,
                             "vc_buf_size": 4
                         },
                         path_pa_db=os.path.join(dir_dsent, "router_pa_db.json"))


def test_router_pa_db():
    path_pa_db = os.path.join(dir_dsent, "router_pa_db.json")
    with open(path_pa_db, "r") as f:
        pa_db_raw = json.load(f)
        nested_dict = lambda: defaultdict(nested_dict)
        pa_db = nested_dict()
        for k_ip in pa_db_raw:
            for k_op in pa_db_raw[k_ip]:
                for k_l in pa_db_raw[k_ip][k_op]:
                    pa_db[int(k_ip)][int(k_op)][float(k_l)] = pa_db_raw[k_ip][k_op][k_l]
    in_port = 5
    out_port = 5
    for _ in trange(10):
        load = 0.5
        assert load >= 0 and load <= 1
        loads = sorted(pa_db[in_port][out_port].keys())
        loads.append(1.1)  # if load is 1
        for i in range(len(loads)):
            if load >= loads[i] and load < loads[i + 1]:
                print(pa_db[in_port][out_port][loads[i]])


def test_eval_p_wire_setup():
    ld = [(1 * 1e-3, 50 * 1e-12)]
    ld += [(1 * 1e-3, 136 * 1e-12), (2 * 1e-3, 209 * 1e-12), (3 * 1e-3, 305 * 1e-12), (4 * 1e-3, 423 * 1e-12),
           (5 * 1e-3, 567 * 1e-12), (6 * 1e-3, 732 * 1e-12), (7 * 1e-3, 921 * 1e-12)]
    ld += [(1 * 1e-3, 1e-9), (2 * 1e-3, 1e-9), (3 * 1e-3, 1e-9), (4 * 1e-3, 1e-9), (5 * 1e-3, 1e-9), (9 * 1e-3, 1e-9),
           (17 * 1e-3, 1e-9)]
    eval_p_wire_setup(dir_dsent=dir_dsent, load_step=0.005, ld=ld, cfg_wire={"process": 45, "freq": 1e9, "width": 128})


def test_eval_P_wire():
    p_db = load_db_wire(os.path.join(dir_dsent, "wire_p_db.json"))
    P = eval_P_wire(p_db=p_db, load=0.5, length=1e-3, delay=143 * 1e-12)
    print(P)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tests for NoC package.")
    parser.add_argument("test_func")
    args = parser.parse_args()
    eval((args.test_func) + "()")