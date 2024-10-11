from collections import defaultdict
import json
from multiprocessing import Pool
import networkx as nx
import numpy as np
import subprocess
import os
import tempfile
import re
import shutil
import matplotlib.pyplot as plt
import math


def _parse_tgff(s):
    """
        s: string of tgff file content
    """
    gs = []  # graphs
    s = s.splitlines()
    idx_g = 0
    while idx_g < len(s):
        if "TASK_GRAPH" in s[idx_g]:  # comes to a new graph
            idx_n = idx_g + 1
            while "ARC" not in s[idx_n]:
                idx_n += 1
            g = nx.DiGraph()
            while "ARC" in s[idx_n]:
                res = re.search("FROM t\d*_(\d*).*t\d*_(\d*)", s[idx_n])
                src = int(res.group(1))
                dst = int(res.group(2))
                g.add_edge(src, dst)
                idx_n += 1
            idx_g = idx_n + 1
            gs.append(g)
        else:
            idx_g += 1
    return gs


def show_graph(G, edge_attr=None):
    """
        show the nx graph using Graphviz
        edge_attr: edge attribute name
    """
    fd, img_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    PG = nx.nx_pydot.to_pydot(G)
    if edge_attr is not None:
        for e in PG.get_edges():
            e.set_label("{}".format(e.get_attributes()[edge_attr]))
    PG.write_png(img_path)
    img = plt.imread(img_path)
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    os.remove(img_path)


def _random_split(n, k, cst):
    """
        Return: rand_list that splits number n to k parts
        cst: constraint list. rand_list[i] <= cst[i]; rand_list[i] >= 1; rand_list[i] is integer 
    """
    assert k == len(cst)
    if n > sum(cst):  # unable to satisfy constraint
        return None
    elif n < k:  # rand_list[i] should >= 1
        return None

    rand_list = []
    for i in range(k):
        high = min(n - sum(rand_list) -
                   (k - 1 - i), cst[i]) + 1  # save 1 for each number after i; +1 for high (exclusive) in randint
        low = max(1, n - sum(cst[i + 1:]) - sum(rand_list))
        r = (n - sum(rand_list)) if (i == k - 1) else np.random.randint(low, high)
        rand_list.append(r)
        assert (r >= 1 and r <= cst[i])
    assert sum(rand_list) == n
    return rand_list


def gen_weight(g, bw, avg_comm, std_comm):
    """
        g: should be nx graph
        bw: unidirectional link bandwidth
        avg_comm: average of comm
        std_comm: standard deviation
    """
    node_num = len(g)
    g_arr = nx.to_numpy_array(g, nodelist=range(node_num), dtype=int, weight=None)

    comm_arr = np.zeros(shape=g_arr.shape, dtype=int)
    for i in range(node_num):
        neighbors = np.where(g_arr[i] > 0)[0]
        if not len(neighbors):  # has no output
            continue
        cst = (bw - comm_arr.sum(axis=0))[neighbors] - g_arr[i + 1:].sum(axis=0)[neighbors]
        min_out = g_arr.sum(axis=1)[i]  # at least 1 for each output
        max_out = min(sum(cst), bw)
        while True:
            assert min_out <= max_out  # this may not hold for large task graph having more than 128 output links
            out = int(np.random.normal(avg_comm, std_comm))
            if out >= min_out and out <= max_out:
                break
        rand_list = _random_split(out, sum(g_arr[i] > 0), cst)
        assert rand_list is not None

        k = 0
        for j in range(node_num):
            if g_arr[i][j]:
                comm_arr[i][j] += rand_list[k]
                k += 1

    assert (comm_arr.sum(axis=0) <= bw).all(), comm_arr.sum(axis=0)
    assert (comm_arr.sum(axis=1) <= bw).all(), comm_arr.sum(axis=1)

    for i in range(node_num):
        for j in range(node_num):
            if g_arr[i][j]:
                assert comm_arr[i][j] > 0
                g.edges[i, j]["comm"] = comm_arr[i][j]


def gen_task_tgff(dir_tgff, bw, cfg=None, filter=None, clean=True):
    """
        Return: list of directed nx task graphs, using "comm" edge attribute as communication rate
        
        bw: the bandwidth of links
        cfg: configuration of tgff
        filter: function to exclude invalid task graph
    """
    dir_tmp = tempfile.mkdtemp(dir=os.path.join(dir_tgff, "build"))
    tgffopt_path = os.path.join(dir_tmp, "tgff")

    if cfg is None:
        cfg = {}
    tg_cnt = cfg["tg_cnt"] if "tg_cnt" in cfg else 5  # number of task graphs to generate
    task_cnt = cfg["task_cnt"] if "task_cnt" in cfg else (20, 1)  # minimum number of tasks per task graph (average, multiplier)
    task_degree = cfg["task_degree"] if "task_degree" in cfg else (3, 3)
    seed = cfg["seed"] if "seed" in cfg else 0

    if filter is None:
        filter = lambda g: len(list(nx.isolates(g))) == 0

    tgffopt_file = ""
    tgffopt_file += "seed {}\n".format(seed)
    tgffopt_file += "tg_cnt {}\n".format(tg_cnt)
    tgffopt_file += "task_cnt {} {}\n".format(task_cnt[0], task_cnt[1])
    tgffopt_file += "period_mul 1\n"
    tgffopt_file += "task_degree {} {}\n".format(task_degree[0], task_degree[1])
    tgffopt_file += "tg_write\n"
    tgffopt_file += "eps_write"

    with open(tgffopt_path + ".tgffopt", "w") as f:
        f.write(tgffopt_file)

    cmd_list = [os.path.join(dir_tgff, "tgff"), tgffopt_path]
    subprocess.run(cmd_list)

    with open(os.path.join(dir_tmp, "tgff.tgff"), "r") as f:
        s = f.read()
        gs = _parse_tgff(s)
    gs = [g for g in gs if filter(g)]  # pick the valid graph

    for g in gs:
        gen_weight(g, bw, avg_comm=bw // 2, std_comm=bw // 6)

    if clean:
        shutil.rmtree(dir_tmp)
    return gs


def mincut_chaco(dir_chaco, G, k, clean=True):
    '''
        get k-mincut by using chaco
        Return: partition list: [[node a, node b, ...], ..., [node x, node y, ...]]
        
        G: networkx graph, the weight name of is "comm"
    '''
    if not nx.is_directed(G):
        raise TypeError("ERROR: unsupported undirected graph")
    if min(G.nodes()) != 0:
        raise ValueError("ERROR: the name of nodes in graph should start from zero")
    if k > len(G) or k < 1:
        raise ValueError("ERROR: invalid number of cut")

    # Note k = 1 must not use chaco, error otherwise
    if k == 1:
        return [list(G.nodes())]

    e_cnt = 0
    graph_file = ''
    for i in G:
        assert not G.has_edge(i, i)  # has not self cycle
        graph_file += '\n1 '  # vertex weight
        for j in G:
            wgt_sum = 0
            wgt_sum += G.edges[(i, j)]["comm"] if G.has_edge(i, j) else 0  # chaco only supports undirected edges
            wgt_sum += G.edges[(j, i)]["comm"] if G.has_edge(j, i) else 0
            if wgt_sum:
                graph_file += '%d %d ' % (j + 1, wgt_sum)  # neighbor edgeweight. node starts from 1
                e_cnt += 1
    graph_file = '%d %d 011' % (len(G), e_cnt / 2) + graph_file  # e_cnt/2: count twice
    dir_tmp = tempfile.mkdtemp(dir=os.path.join(dir_chaco, "build"))
    graph_path = os.path.join(dir_tmp, "graph.graph")
    res_path = os.path.join(dir_tmp, "graph.res")
    with open(graph_path, "w") as f:
        f.write(graph_file)

    cmd_input = graph_path + '\n' + res_path + '\n' + '2\n2\n1\n' + str(
        k) + '\n2' + '\nn\n'  # consider 'Partitioning dimension'
    cmd_list = ["./chaco"]
    run_res = subprocess.run(cmd_list, capture_output=True, text=True, input=cmd_input, cwd=dir_chaco)
    try:
        with open(res_path, 'r') as r_f:
            res_raw = r_f.readlines()
            res = []
            for l in res_raw:
                if l[0:2] == '  ':
                    res.append([])
                else:
                    res[-1].append(int(l) - 1)
    except:  # something goes wrong
        raise RuntimeError("ERROR: Chaco runs failed!!!", "|||STDOUT:", run_res.stdout, "|||STDERR:", run_res.stderr)
    if clean:
        shutil.rmtree(dir_tmp)
    return res


def check_deadlock(rtp):
    """
        Return: True for passing the deadlock checking
        rtp: routing path list dict; rtp[(src node, dst node)] = [rtr0, rtr1, ...]
    """
    LDG = nx.DiGraph()  # link dependency graph
    for (src, dst) in rtp:
        assert len(rtp[(src, dst)]) > 0  # nodes can not be connected without router
        path = [src] + rtp[(src, dst)] + [dst]
        links = list(zip(path, path[1:]))  # [(src, rtr0), (rtr0, rtr1), ..., (rtrn, dst)]
        LDG.add_edges_from(zip(links, links[1:]))
    return nx.is_directed_acyclic_graph(LDG)


def _get_TPS(G, TPS, special_node=None):
    '''
        find the Turn Prohibition Set
        G: undirected nx graph
        TPS: list of turn(tuple)
    '''
    assert len(G) > 0
    if len(G) == 1:
        return  # If there is only one remaining node in G then delete it and return the procedure.

    # Select the node of minimal degree in G, excluding the special node (if there is such one).
    n_degree = min([d if n != special_node else float('inf') for (n, d) in G.degree])

    #  If several nodes of minimal degree are available,
    # then select first a node that is not a neighbor of the special node
    # assert nx.is_connected(G)
    node_a = [n for n in G if G.degree[n] == n_degree if n != special_node]
    if len(node_a) > 1:
        node_a_not_special_node_neigh = [n_a for n_a in node_a if special_node not in G.neighbors(n_a)]
        if node_a_not_special_node_neigh:
            node_a = node_a_not_special_node_neigh[0]
        else:  # all connected to special node
            node_a = node_a[0]
    else:
        node_a = node_a[0]

    # Prohibit all the turns around node a, that is, prohibit all the turns of the type (b, a, c)
    for i in G.neighbors(node_a):
        for j in G.neighbors(node_a):
            if i < j:
                TPS.append((i, node_a, j))

    # Permit all the turns starting from node a, that is, permit all the turns of the type (a, b, c)
    for t in TPS[:]:  # you can not del while iterating ('for ... in' still use idx) unless copy the list.
        if (t[0] in G and t[1] in G and t[2] in G) and (t[0] == node_a or t[2] == node_a):  # undireted
            print('warning Permit')
            TPS.remove(t)

    G_ = G.copy()
    G_.remove_node(node_a)
    if nx.number_connected_components(G_) == 1:
        # print('delete node_a : {} G1 is {}'.format(node_a, G_.nodes))
        _get_TPS(G_, TPS, special_node=special_node)
    else:
        # print('delete node_a : {}'.format(node_a))
        # If the remaining graph is broken into K  components of connectivity G1, G2, . . . , GK,
        # then select K special links connecting node a to each component
        connected_components = list(nx.connected_components(G_))
        special_link_node = [None] * len(connected_components)
        for idx_c, c in enumerate(connected_components):  # asign every component a link
            for n in G.neighbors(node_a):
                if n in c and special_link_node[idx_c] is None:
                    special_link_node[idx_c] = n
        # print('special_link_node is {}'.format(special_link_node))
        for t in TPS[:]:  # permit all the turns between the special links
            if t[0] in special_link_node and t[2] in special_link_node and t[1] == node_a:
                TPS.remove(t)

        sub_G = []
        for c in connected_components:
            g = nx.Graph()
            g.add_nodes_from(c)
            g.add_edges_from((u, v) for (u, v) in G.edges() if u in g if v in g)
            sub_G.append(g)
            # print(g.nodes)
        # If a special node exists in G, then it should be in G1.
        # For each of the other components of connectivity G2, G3, . . . , GK,
        # the node connected to the special link is marked as a special node.
        if special_node is not None:
            id_special_node_components = [
                i for i in range(len(connected_components)) if special_node in connected_components[i]
            ][0]  # where is special node
            for idx_g, g in enumerate(sub_G):
                if idx_g == id_special_node_components:
                    _get_TPS(g, TPS, special_node=special_node)
                else:
                    _get_TPS(g, TPS, special_node=special_link_node[idx_g])
        else:
            for idx_g, g in enumerate(sub_G):
                if idx_g == 0:
                    _get_TPS(g, TPS, special_node=None)  # require only K - 1 special nodes
                else:
                    _get_TPS(g, TPS, special_node=special_link_node[idx_g])


def get_TPS(G):
    """
        Return: prohibitive turn set of undirected graph G; [(a, b, c), (d, e, g)...]
    """
    if nx.is_directed(G):
        raise TypeError("ERROR: G should be undirected graph")
    TPS = []
    _get_TPS(G, TPS)
    for i in G.nodes():
        for j in G.neighbors(i):
            TPS.append((j, i, j))
    return TPS


def routing_base(task_graph, topo_graph, TPS, bw, weight_func=None):
    """
        Return: routing path list dict; used algorithm: https://www.docin.com/p-725545101.html
        task_graph: directed nx graph; node name should be 0 ~ (core_num - 1); use "comm" field to store traffic rate in edge attributes
        topo_graph: directed nx graph; node name should be 0 ~ (node_num - 1)
        comm_pair: list of tuple; [(src0, dst0), ...]
        bw: bandwidth of links
        weight_func: lambda function, used for updating edge weight in shortest path routing
            parameter: acc_comm: accumulative traffic on the link, cur_comm: traffic of current considered flow 
    """
    if weight_func is None:  # should be > 0 for dijkstra algo
        weight_func = lambda acc_comm, cur_comm: 1 if (acc_comm + cur_comm) <= bw else (acc_comm + cur_comm - bw + 1) * 100

    if not nx.is_directed(task_graph) or not nx.is_directed(topo_graph):
        raise TypeError("ERROR: task/topology graph should be directed")
    assert len(task_graph) < len(topo_graph)  # have at least one router

    tg = topo_graph.copy()
    virt_src_edges = []
    virt_dst_edges = []
    comm_pair = list(task_graph.edges(data=True))
    comm_pair = sorted(comm_pair, key=lambda cp: cp[2]["comm"], reverse=True)  # sorted the traffic flow,
    # add aux edges
    for (s, d, _) in comm_pair:
        virt_src = len(tg)
        virt_dst = len(tg) + 1
        src_edge = (virt_src, s)
        dst_edge = (d, virt_dst)
        tg.add_edge(*src_edge)
        tg.add_edge(*dst_edge)
        virt_src_edges.append(src_edge)
        virt_dst_edges.append(dst_edge)

    H = nx.line_graph(tg)

    # remove prohibited turn
    for (a, b, c) in TPS:
        e_m = (a, b)
        e_n = (b, c)
        e_m_r = (b, a)
        e_n_r = (c, b)

        if not H.has_edge(e_m, e_n) and not H.has_edge(e_n_r, e_m_r):
            raise ValueError("ERROR: prohibited turn not used, something must be wrong")
        if H.has_edge(e_m, e_n):
            H.remove_edge(e_m, e_n)
        if H.has_edge(e_n_r, e_m_r):
            H.remove_edge(e_n_r, e_m_r)

    nx.set_edge_attributes(H, 0, "acc_comm")
    nx.set_edge_attributes(H, weight_func(0, 0), "w")  # initial weight, accumulated comm is 0

    rtp = {}
    for idx, (s, d, edata) in enumerate(comm_pair):
        for e_m, e_n in H.edges():
            H.edges[(e_m, e_n)]["w"] = weight_func(H.edges[(e_m, e_n)]["acc_comm"], edata["comm"])

        p = nx.shortest_path(H, source=virt_src_edges[idx], target=virt_dst_edges[idx], weight="w")
        for e_m in p:  # update the edge weight according to accumulated communication
            for e_n in H.successors(e_m):
                H.edges[(e_m, e_n)]["acc_comm"] += edata["comm"]

        p = p[:-1]  # remove dst_edge
        p = [pp[1] for pp in p]
        rtp[(s, d)] = p[1:-1]  # remove src, dst
    return rtp


def eval_PPA_booksim(dir_booksim, task_graph, topo_graph, rtp, cfg=None, clean=True):
    """
        Return: dict, keys are power/latency/area for 32nm process
        task_graph: directed graph, the node name should be 0 ~ (core number - 1)
        topo_graph: directed graph, the node name should be 0 ~ (core number + router number - 1)
        rtp: routing path list dict: rtp[(src, dst)] = [rtr_0, rtr_1, ...]
        cfg: a bunch of NoC parameters, including "packet_size", "timeout", "clk_freq", "channel_width"
    """
    core_num = len(task_graph)

    if not nx.is_directed(task_graph) or not nx.is_directed(topo_graph):
        raise ValueError("ERROR: task/topology graph should be directed")

    assert set(task_graph.nodes()) == set(range(len(task_graph)))
    assert set(topo_graph.nodes()) == set(range(len(topo_graph)))  # node name check

    # topology rule check
    for n in topo_graph:
        if n < core_num:  # is a core
            neighbor = set.union(set(topo_graph.predecessors(n)), set(topo_graph.successors(n)))
            if len(neighbor) != 1:
                raise ValueError("ERROR: core {} connects to 0 or more than 1 core/router:{}".format(n, topo_graph.edges()))
            neighbor = neighbor.pop()
            if neighbor < core_num:
                raise ValueError("ERROR: core {} connects to core instead of router:{}".format(n, topo_graph.edges()))
        else:  # is a router
            if topo_graph.in_degree(n) == 0 or topo_graph.out_degree(n) == 0:
                raise ValueError("ERROR: router {} has only input/output links:{}".format(n, topo_graph.edges()))

    # routing path check
    if set(task_graph.edges()) != set(rtp):
        raise ValueError("ERROR: communication pairs in task graph and rtp not equal, {}, {}".format(task_graph.edges(), rtp))

    dir_tmp = tempfile.mkdtemp(dir=os.path.join(dir_booksim, "build"))
    cfg_tpl_path = os.path.join(dir_booksim, "config.template")
    cfg_path = os.path.join(dir_tmp, "config")
    netfile_path = os.path.join(dir_tmp, "netfile")
    trace_path = os.path.join(dir_tmp, "trace")
    techfile_path = os.path.join(dir_booksim, "techfile.txt")

    if cfg is None:
        cfg = {}
    packet_size = cfg["packet_size"] if "packet_size" in cfg else 8  # unit: flit
    timeout = cfg["timeout"] if "timeout" in cfg else 60  # timeout of running Booksim
    clk_freq = cfg["clk_freq"] if "clk_freq" in cfg else 1  # unit: Ghz
    assert clk_freq <= 3  # this is Ghz, avoid stupid mistake
    channel_width = cfg["channel_width"] if "channel_width" in cfg else 128  # unit: bit
    sim_cycle = cfg["sim_cycle"] if "sim_cycle" in cfg else 10000  # duration of packet injection
    num_vcs = cfg["num_vcs"] if "num_vcs" in cfg else 1
    vc_buf_size = cfg["vc_buf_size"] if "vc_buf_size" in cfg else 4

    with open(cfg_tpl_path, "r") as f_cfg_tpl, open(cfg_path, "w") as f_cfg:
        cfg_tpl = f_cfg_tpl.read()
        cfg_tpl = cfg_tpl.replace("*netfile_path*", netfile_path)
        cfg_tpl = cfg_tpl.replace("*techfile_path*", techfile_path)
        cfg_tpl = cfg_tpl.replace("*channel_width*", str(channel_width))
        cfg_tpl = cfg_tpl.replace("*num_vcs*", str(num_vcs))
        cfg_tpl = cfg_tpl.replace("*vc_buf_size*", str(vc_buf_size))
        f_cfg.write(cfg_tpl)

    topo_graph_ud = nx.to_undirected(topo_graph)  # undirected
    # for the bug in booksim readFile(), which is bi-directional connection
    with open(netfile_path, "w") as f_netfile:
        netfile = ""
        for i in topo_graph_ud:
            if i >= core_num:
                netfile += "node {} ".format(i) if i < core_num else "router {} ".format(
                    i - core_num)  # router starts from zero in booksim
                for j in topo_graph_ud.neighbors(i):
                    netfile += "node {} ".format(j) if j < core_num else "router {} ".format(j - core_num)
                netfile += "\n"

        for (src, dst) in rtp:
            netfile += "* {} {} ".format(src, dst)
            netfile += " ".join(map(lambda e: str(e - core_num), rtp[(src, dst)]))
            netfile += "\n"

        f_netfile.write(netfile)

    # generate tace file according to poisson/exponential distribution
    # generate for every comm pair independently and then conbine them
    flag_wrap = False
    trace_dict = {}  # trace_dict[(src, dst)] = [x, y, z, ...]
    for (src, dst) in rtp:
        trace_dict[(src, dst)] = np.zeros(sim_cycle, int)
        traffic_rate = task_graph.edges[(src, dst)]["comm"]  # unit: GBit / s
        lambda_exp = traffic_rate / (packet_size * channel_width * clk_freq)  # exponential distribution
        cnt_cyc = 0
        while cnt_cyc < sim_cycle:
            cnt_cyc += round(1 /
                             lambda_exp)  # round(np.random.exponential(1 / lambda_exp)), change Poisson to normal distribution
            if cnt_cyc < sim_cycle:
                trace_dict[(src, dst)][cnt_cyc] = 1

    with open(trace_path, "w") as f_trace:
        last_cyc = 0
        for c in range(sim_cycle):
            for (src, dst) in trace_dict:
                if trace_dict[(src, dst)][c]:
                    if flag_wrap:
                        f_trace.write("\n")  # be careful with redundant wrap
                    else:
                        flag_wrap = True
                    delay = c - last_cyc
                    last_cyc = c
                    f_trace.write("{} {} {} 0".format(delay, src, dst))

    cmd_list = [
        os.path.join(dir_booksim, "booksim"), cfg_path, "-",
        "workload=trace({" + trace_path + "},{" + str(packet_size) + "," + str(packet_size) + "},-1})"
    ]
    run_res = subprocess.run(cmd_list, capture_output=True, text=True, timeout=timeout)  # timeout for deadlock
    if run_res.returncode != 0:
        raise RuntimeError("ERROR: booksim exits error with: {}".format(run_res.stderr))

    run_output = run_res.stdout
    # parse the output result
    re_dict = {
        "avg_hop": "(?:.|\n)*Overall average hops =(.*)\(",
        "avg_lat_packet": "(?:.|\n)*Overall average packet latency =(.*)\(",
        "avg_lat_network": "(?:.|\n)*Overall average network latency =(.*)\(",
        "power": "(?:.|\n)*Total Power:(.*)\n",
        "area": "(?:.|\n)*Total Area:(.*)\n"
    }
    PPA = {}
    for k in re_dict:
        v = re.search(re_dict[k], run_output)
        if v is None:
            raise ValueError("ERROR: {} not found, and output is:{}".format(k, run_output))
        PPA[k] = float(v.group(1))

    if clean:
        shutil.rmtree(dir_tmp)

    if "network deadlock" in run_output:
        raise RuntimeError(run_output)
    return PPA


def eval_router_dsent(dir_dsent, in_port, out_port, load, cfg=None, clean=True):
    """
        Use dsent to get power & area of a single router
        in/out_port: number of input/output ports
        load: average load of all input channels, 
            that is (load of input channel 1, load of input channel 2, ...) / number of input channels
        cfg: dict, keys are process(22/32/45), freq(xx Hz), channel_width(xx bit), 
            num_vc, vc_buf_size
    """
    if in_port <= 0 or out_port <= 0:
        raise ValueError("ERROR: invalid port number")

    path_cfg_tpl = os.path.join(dir_dsent, "router.cfg.template")
    dir_tmp = tempfile.mkdtemp(dir=os.path.join(dir_dsent, "build"))
    path_cfg = os.path.join(dir_tmp, "router.cfg")

    if cfg is None:
        cfg = {}

    process = cfg["process"] if "process" in cfg else 32
    if process != 22 and process != 32 and process != 45:
        raise ValueError("ERROR: Dsent has illegal process {}".format(process))
    freq = cfg["freq"] if "freq" in cfg else 1e9  # Hz
    channel_width = cfg["channel_width"] if "channel_width" in cfg else 128
    num_vc = cfg["num_vc"] if "num_vc" in cfg else 4
    vc_buf_size = cfg["vc_buf_size"] if "vc_buf_size" in cfg else 4

    with open(path_cfg_tpl, "r") as f_cfg_tpl:
        cfg_tpl = f_cfg_tpl.read()
        cfg_tpl = cfg_tpl.replace("*InjectionRate*", str(load))
        cfg_tpl = cfg_tpl.replace("*ElectricalTechModelFilename*",
                                  os.path.join(dir_dsent, "tech_models/Bulk{}LVT.model".format(process)))
        cfg_tpl = cfg_tpl.replace("*Frequency*", str(freq))
        cfg_tpl = cfg_tpl.replace("*NumberInputPorts*", str(in_port))
        cfg_tpl = cfg_tpl.replace("*NumberOutputPorts*", str(out_port))
        cfg_tpl = cfg_tpl.replace("*NumberBitsPerFlit*", str(channel_width))
        cfg_tpl = cfg_tpl.replace("*NumberVirtualChannelsPerVirtualNetwork*", str(num_vc))
        cfg_tpl = cfg_tpl.replace("*NumberBuffersPerVirtualChannel*", str(vc_buf_size))

    with open(path_cfg, "w") as f_cfg:
        f_cfg.write(cfg_tpl)

    cmd_list = [os.path.join(dir_dsent, "dsent"), "-cfg", path_cfg]
    run_res = subprocess.run(cmd_list, capture_output=True, text=True)
    if run_res.returncode != 0:
        raise RuntimeError(run_res.stderr)
    run_output = run_res.stdout

    # parse the output result
    re_dict = {
        "buffer_dynamic": "(?:.|\n)*Buffer dynamic power:(.*)\n",
        "buffer_leakage": "(?:.|\n)*Buffer leakage power:(.*)\n",
        "xbar_dynamic": "(?:.|\n)*Crossbar dynamic power:(.*)\n",
        "xbar_leakage": "(?:.|\n)*Crossbar leakage power:(.*)\n",
        "sa_dynamic": "(?:.|\n)*Switch allocator dynamic power:(.*)\n",
        "sa_leakage": "(?:.|\n)*Switch allocator leakage power:(.*)\n",
        "clock_dynamic": "(?:.|\n)*Clock dynamic power:(.*)\n",
        "clock_leakage": "(?:.|\n)*Clock leakage power:(.*)\n",
        "total_dynamic": "(?:.|\n)*Total dynamic power:(.*)\n",
        "total_leakage": "(?:.|\n)*Total leakage power:(.*)\n",
        "buf_area": "(?:.|\n)*Buffer area:(.*)\n",
        "xbar_area": "(?:.|\n)*Crossbar area:(.*)\n",
        "sa_area": "(?:.|\n)*Switch allocator area:(.*)\n",
        "total_area": "(?:.|\n)*Total area:(.*)\n",
        "total_power": "(?:.|\n)*Total power:(.*)\n"
    }
    PA = {}
    for k in re_dict:
        v = re.search(re_dict[k], run_output)
        if v is None:
            raise ValueError("ERROR: {} not found, and output is:{}".format(k, run_output))
        PA[k] = float(v.group(1))

    if clean:
        shutil.rmtree(dir_tmp)

    return PA


def eval_wire_dsent(dir_dsent, load, cfg=None, clean=True):
    """
        return power of wire using dsent
    """
    path_cfg_tpl = os.path.join(dir_dsent, "wire.cfg.template")
    dir_tmp = tempfile.mkdtemp(dir=os.path.join(dir_dsent, "build"))
    path_cfg = os.path.join(dir_tmp, "wire.cfg")

    if cfg is None:
        cfg = {}

    process = cfg["process"] if "process" in cfg else 32
    if process != 22 and process != 32 and process != 45:
        raise ValueError("ERROR: Dsent has illegal process {}".format(process))
    freq = cfg["freq"] if "freq" in cfg else 1e9  # Hz
    width = cfg["width"] if "width" in cfg else 128
    length = cfg["length"] if "length" in cfg else 1e-3  # unit: meter
    delay = cfg["delay"] if "delay" in cfg else (1 / freq)  # may be smaller than 1.0 / Frequency

    with open(path_cfg_tpl, "r") as f_cfg_tpl:
        cfg_tpl = f_cfg_tpl.read()
        cfg_tpl = cfg_tpl.replace("*InjectionRate*", str(load))
        cfg_tpl = cfg_tpl.replace("*ElectricalTechModelFilename*",
                                  os.path.join(dir_dsent, "tech_models/Bulk{}LVT.model".format(process)))
        cfg_tpl = cfg_tpl.replace("*Frequency*", str(freq))
        cfg_tpl = cfg_tpl.replace("*NumberBits*", str(width))
        cfg_tpl = cfg_tpl.replace("*WireLength*", str(length))
        cfg_tpl = cfg_tpl.replace("*Delay*", str(delay))

    with open(path_cfg, "w") as f_cfg:
        f_cfg.write(cfg_tpl)

    cmd_list = [os.path.join(dir_dsent, "dsent"), "-cfg", path_cfg]
    run_res = subprocess.run(cmd_list, capture_output=True, text=True)
    if run_res.returncode != 0:
        raise RuntimeError(run_res.stderr)
    run_output = run_res.stdout

    re_dict = {"dynamic": "(?:.|\n)*Dynamic power:(.*)\n", "leakage": "(?:.|\n)*Leakage power:(.*)\n"}
    P = {}
    for k in re_dict:
        v = re.search(re_dict[k], run_output)
        if v is None:
            raise ValueError("ERROR: {} not found, and output is:{}".format(k, run_output))
        P[k] = float(v.group(1))

    if clean:
        shutil.rmtree(dir_tmp)

    return P


def eval_pa_router_setup(dir_dsent, port_range, load_step, processes, cfg_router, path_pa_db=None):
    """
        Traverse the combinations and store.
        dir_dsent: absolute path of dsent directory
        path_pa_db: absolute path to store the power database which is json format
        port_range: the range of port number, iterator
        load_step: the step of traversing load
    """
    if path_pa_db is None:
        path_pa_db = os.path.join(dir_dsent, "router_pa_db.json")

    nested_dict = lambda: defaultdict(nested_dict)
    pa_db = nested_dict()
    res = nested_dict()
    # use multiprocess
    pool = Pool(processes=processes)
    for in_port in port_range:
        for out_port in port_range:
            for load in np.arange(0, 1, load_step):
                res[in_port][out_port][load] = pool.apply_async(eval_router_dsent,
                                                                kwds=dict(dir_dsent=dir_dsent,
                                                                          in_port=in_port,
                                                                          out_port=out_port,
                                                                          load=load,
                                                                          cfg=cfg_router,
                                                                          clean=True))
    pool.close()
    pool.join()
    for in_port in port_range:
        for out_port in port_range:
            for load in np.arange(0, 1, load_step):
                pa_db[in_port][out_port][load] = res[in_port][out_port][load].get()
    with open(path_pa_db, "w") as f:
        json.dump(pa_db, f, indent=4)


def eval_p_wire_setup(dir_dsent, load_step, ld, cfg_wire, path_p_db=None):
    """
        Generate the power database of wire, key is [length][delay][load]
        ld: list of (length, delay),unit of length is meter, unit of delay is second
    """
    if path_p_db is None:
        path_p_db = os.path.join(dir_dsent, "wire_p_db.json")

    assert "length" not in cfg_wire
    assert "delay" not in cfg_wire

    nested_dict = lambda: defaultdict(nested_dict)
    p_db = nested_dict()
    for (length, delay) in ld:
        assert length <= 0.02 and delay <= 1e-9
        for load in np.arange(0, 1, load_step):
            cfg_wire["length"] = length
            cfg_wire["delay"] = delay
            p_db[length][delay][load] = eval_wire_dsent(dir_dsent=dir_dsent, load=load, cfg=cfg_wire, clean=True)
    with open(path_p_db, "w") as f:
        json.dump(p_db, f, indent=4)


def load_db_router(path_pa_db):
    with open(path_pa_db, "r") as f:
        pa_db_raw = json.load(f)
        nested_dict = lambda: defaultdict(nested_dict)
        pa_db = nested_dict()
        for k_ip in pa_db_raw:
            for k_op in pa_db_raw[k_ip]:
                for k_l in pa_db_raw[k_ip][k_op]:
                    pa_db[int(k_ip)][int(k_op)][float(k_l)] = pa_db_raw[k_ip][k_op][k_l]
    return pa_db


def load_db_wire(path_p_db):
    """
        load the database
    """
    with open(path_p_db, "r") as f:
        p_db_raw = json.load(f)
    nested_dict = lambda: defaultdict(nested_dict)
    p_db = nested_dict()
    for k_len in p_db_raw:
        for k_d in p_db_raw[k_len]:
            for k_l in p_db_raw[k_len][k_d]:
                p_db[float(k_len)][float(k_d)][float(k_l)] = p_db_raw[k_len][k_d][k_l]
    return p_db


def eval_PA_router(pa_db, in_port, out_port, load):
    assert load >= 0 and load <= 1
    in_port = min(in_port, 8)  # TODO: SiP issue
    out_port = min(out_port, 8)
    loads = sorted(pa_db[in_port][out_port].keys())
    loads.append(1.1)  # if load is 1
    for i in range(len(loads)):
        if load >= loads[i] and load < loads[i + 1]:
            return pa_db[in_port][out_port][loads[i]]
    print("ERROR: ***, ", in_port, out_port, load)


def eval_P_wire(p_db, load, length, delay):
    assert load >= 0 and load <= 1
    for k_len in p_db:
        if math.isclose(k_len, length):
            for k_d in p_db[k_len]:
                if math.isclose(k_d, delay):
                    loads = sorted(p_db[k_len][k_d].keys())
                    loads.append(1.1)  # if load is 1
                    for i in range(len(loads)):
                        if load >= loads[i] and load < loads[i + 1]:
                            return p_db[k_len][k_d][loads[i]]
    raise ValueError("ERROR: wire data not found. load:{}, length:{}, delay:{}".format(load, length, delay))