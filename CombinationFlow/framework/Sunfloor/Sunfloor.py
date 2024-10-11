import os
import numpy as np
from multiprocessing import Pool
from ..NoC import mincut_chaco, get_TPS, eval_PA_router, load_db_router
import networkx as nx
import copy


def routing(src, dst, SCG, TPS):
    """
        Return: routing path: [src, ... , dst]; used algorithm: https://www.docin.com/p-725545101.html
        bw: bandwidth of links
    """
    scg = SCG.copy()  # independent shallow copy
    # remove ban edges
    for (u, v, eattr) in SCG.edges(data=True):
        if eattr["ban"]:
            scg.remove_edge(u, v)

    virt_src = len(scg)
    virt_dst = len(scg) + 1
    src_edge = (virt_src, src)
    dst_edge = (dst, virt_dst)
    scg.add_edge(*src_edge)
    scg.add_edge(*dst_edge)
    H = nx.line_graph(scg)

    # remove prohibited turn
    for (a, b, c) in TPS:
        e_m = (a, b)
        e_n = (b, c)
        e_m_r = (b, a)
        e_n_r = (c, b)
        if H.has_edge(e_m, e_n):  # some turn in TPS might be deleted for ban = True
            H.remove_edge(e_m, e_n)
        if H.has_edge(e_n_r, e_m_r):
            H.remove_edge(e_n_r, e_m_r)

    # add weight to H
    nx.set_edge_attributes(H, 0, name="w")
    for (u, v, eattr) in scg.edges(data=True):
        e_m = (u, v)
        for e_n in H.successors((u, v)):
            H.edges[(e_m, e_n)]["w"] += eattr["weight"] if e_m != src_edge else 0
    try:
        p = nx.shortest_path(H, source=src_edge, target=dst_edge, weight="w")
    except nx.NetworkXNoPath:
        return None
    else:
        p = p[:-1]  # remove virtual dst_edge
        p = [pp[1] for pp in p]
        return p


def margin_cost(pa_db, SCG, u, v, l_f):
    """
        For reuse opened links, just need to evaluate the cost of increasing load of router v.
            For open new link, we do not know the change of input port in u and output port in v,
            so assume them unchanged
        u, v: src/dst router of link
        l_f: the considered flow / bandwidth
    """
    assert l_f >= 0
    if SCG[u][v]["PHY"]:  # reuse existing links
        assert "r_{}".format(u) in SCG.nodes[v]["switching_activity"]
        loads_v = list(SCG.nodes[v]["switching_activity"].values())
        old_load = sum(loads_v) / len(loads_v)
        new_load = (sum(loads_v) + l_f) / len(loads_v)

        old = eval_PA_router(pa_db=pa_db,
                             in_port=SCG.nodes[v]["switch_size_in"],
                             out_port=SCG.nodes[v]["switch_size_out"],
                             load=old_load)
        new = eval_PA_router(pa_db=pa_db,
                             in_port=SCG.nodes[v]["switch_size_in"],
                             out_port=SCG.nodes[v]["switch_size_out"],
                             load=new_load)
        cost = new["total_power"] - old["total_power"]
        assert cost > 0
    else:  # open&use
        assert "r_{}".format(u) not in SCG.nodes[v]["switching_activity"]
        load_u = list(SCG.nodes[u]["switching_activity"].values())
        loads_v = list(
            SCG.nodes[v]
            ["switching_activity"].values())  # note that loads_v might be [] for the partition only has input traffic of cores

        u_switch_size_in = SCG.nodes[u]["switch_size_in"] if SCG.nodes[u]["switch_size_in"] else 1
        u_switch_size_out = SCG.nodes[u]["switch_size_out"] if SCG.nodes[u]["switch_size_out"] else 1
        v_switch_size_in = SCG.nodes[v]["switch_size_in"] if SCG.nodes[v]["switch_size_in"] else 1
        v_switch_size_out = SCG.nodes[v]["switch_size_out"] if SCG.nodes[v]["switch_size_out"] else 1

        old_u = eval_PA_router(pa_db=pa_db,
                               in_port=u_switch_size_in,
                               out_port=u_switch_size_out,
                               load=np.average(load_u) if len(load_u) else 0)

        new_u = eval_PA_router(pa_db=pa_db,
                               in_port=u_switch_size_in,
                               out_port=u_switch_size_out + 1,
                               load=np.average(load_u) if len(load_u) else 0)  # u opens an output channel

        old_v = eval_PA_router(pa_db=pa_db,
                               in_port=v_switch_size_in,
                               out_port=v_switch_size_out,
                               load=np.average(loads_v) if len(loads_v) else 0)

        new_v = eval_PA_router(pa_db=pa_db,
                               in_port=v_switch_size_in + 1,
                               out_port=v_switch_size_out,
                               load=np.average(loads_v + [l_f]))  # v opens an input channel
        cost = (new_u["total_power"] - old_u["total_power"]) + (new_v["total_power"] - old_v["total_power"]
                                                                )  # cost of add output port + cost of add input port
        assert cost > 0
    return cost


def margin_cost_fixed(pa_db, SCG, u, v, l_f, fixed_port=5):
    """
        The size/port number is fixed for general interposer
        u, v: src/dst router of link
        l_f: the considered flow / bandwidth
    """
    pass


def path_compute(pa_db, i, max_port, bw, SCG, TPS, comm_matrix):
    """
        Algorithm 2. PATH_COMPUTE(i, SCG, rho, PTS, theta), return None for failure, otherwise return routing path list dict
        i: number of partitions
    """
    scg = copy.deepcopy(SCG)

    rtp = {}  # use (src, dst) as key
    # 1. Initialize the set PHY(i1,j1) to false and Bw_avail(i1,j1)
    nx.set_edge_attributes(scg, False, name="PHY")  # PHY is True for links used
    nx.set_edge_attributes(scg, bw, name="Bw_avail")
    nx.set_edge_attributes(scg, 0, name="weight")

    for (src, dst), f in sorted(np.ndenumerate(comm_matrix), key=lambda k: k[1],
                                reverse=True):  # 3. for each flow fk, k=1 ... |F| in decreasing order of flows
        if f == 0:
            continue
        nx.set_edge_attributes(scg, False, "ban")  # ban the link for 'port number' or 'traffic' exceed
        for i1, j1 in [(i1, j1) for i1 in range(i) for j1 in range(i) if i1 != j1]:  # traverse the fully connected graph
            eattr1 = scg[i1][j1]
            if (eattr1["PHY"] and eattr1["Bw_avail"] < (f + 20)) or (not eattr1["PHY"] and (
                (scg.nodes[i1]["switch_size_out"] >= max_port or scg.nodes[j1]["switch_size_in"] >= max_port))):  # TODO: eattr1["Bw_avail"] < f
                eattr1["ban"] = True
            else:
                eattr1["weight"] = margin_cost(pa_db, scg, i1, j1,
                                               f / bw)  # 14. Assign cost(i1, j1) to the edge W(i1, j1) in SCG

        # 15. Find the least cost path between the partitions in which source (sk) and destination (dk) of the flow are present inthe SCG.
        # Choose only those paths that have turns not prohibited by PTS
        p = routing(src, dst, scg, TPS)
        if p is None:
            return None  # failure

        # 16. Update PHY, Bw_avail, switch_size_in, switch_size_out, switching_activity for chosen path
        for i_p, j_p in zip(p, p[1:]):
            scg.edges[(i_p, j_p)]["Bw_avail"] -= f
            if not scg.edges[(i_p, j_p)]["PHY"]:
                scg.edges[(i_p, j_p)]["PHY"] = True
                scg.nodes[i_p]["switch_size_out"] += 1
                scg.nodes[j_p]["switch_size_in"] += 1
                scg.nodes[j_p]["switching_activity"]["r_{}".format(
                    i_p)] = f / bw  # switching activity of the router connected to the source core has been modified

        rtp[(src, dst)] = p[1:-1]  # remove src/dst
    return rtp


def check_deadlock(rtp):
    """
        Return: True for passing the deadlock checking
        rtp: routing path list dict; rtp[(src node, dst node)] = [rtr0, rtr1, ...]
    """
    LDG = nx.DiGraph()  # link dependency graph
    for (src, dst) in rtp:
        path = [src] + rtp[(src, dst)] + [dst]
        links = list(zip(path, path[1:]))  # [(src, rtr0), (rtr0, rtr1), ..., (rtrn, dst)]
        LDG.add_edges_from(zip(links, links[1:]))
    return nx.is_directed_acyclic_graph(LDG)


def sunfloor(dir_chaco, dir_dsent, task_graph, max_port, bw):
    """
        Algorithm 1. Topolgoy Design Algorithm.
        Return: topology graph: nx directed graph, routing path: {(src 0, dst 0):[]} 
        
        task_graph: nx.Digraph, use "comm" as traffic rate. Name of nodes must be 0, 1, 2...
        max_port: number of router max_input/output ports
        bw: channel bandwidth
    """
    if not nx.is_directed(task_graph):
        raise TypeError("ERROR: task graph should be directed")

    path_pa_db = os.path.join(dir_dsent, "router_pa_db.json")
    if not os.path.exists(path_pa_db):
        raise ValueError("ERROR: power/area database not exists")
    pa_db = load_db_router(path_pa_db)

    # check the bw constraint of task_graph
    for n in task_graph:
        if nx.is_isolate(task_graph, n):
            raise ValueError("ERROR: node {} is task graph is isolated".format(n))

        in_traffic = 0
        for (_, _, attr) in task_graph.in_edges(n, data=True):
            in_traffic += attr["comm"]
        if in_traffic > bw:
            raise ValueError("ERROR: node {} in the task graph has too much traffic {}".format(n, in_traffic))

        out_traffic = 0
        for (_, _, eattr) in task_graph.out_edges(n, data=True):
            out_traffic += eattr["comm"]
        if out_traffic > bw:
            raise ValueError("ERROR: node {} in the task graph has too much traffic {}".format(n, out_traffic))

    V = len(task_graph)  # number of nodes
    for i in range(1, V + 1):
        partitions = mincut_chaco(dir_chaco, task_graph, i)  # 3. Find i min-cut partitions of the core graph

        SCG = nx.DiGraph()  # node 0, 1, ... is actually router V, V+1, ...
        # 4. Establish a switch with Nj inputs and outputs for each partition, Nj is the number of vertices (cores) in partition i
        # Note Algorithm 2.2 also say: Initialize switch_size_in(j) and switch_size_out(j) to Nj;
        #     Find switching_activity(j) for each switch, based on the traffic flow within the partition.
        if i == 1:
            SCG.add_node(0)
        else:
            SCG.add_edges_from([(i1, j1) for i1 in range(i) for j1 in range(i) if i1 != j1])
        nx.set_node_attributes(SCG, 0, "switch_size_in")
        nx.set_node_attributes(SCG, 0, "switch_size_out")
        nx.set_node_attributes(SCG, dict([(n, {}) for n in SCG]),
                               "switching_activity")  # injection rate(flits per cycle) of each port
        for idx_pt, pt in enumerate(partitions):
            for c in pt:  # connect cores within the partition to same router
                if task_graph.in_degree(c):  # core has input traffic
                    SCG.nodes[idx_pt]["switch_size_out"] += 1

                if task_graph.out_degree(c):
                    SCG.nodes[idx_pt]["switch_size_in"] += 1

                    out_traffic = 0  # output traffic of router
                    for (_, _, eattr) in task_graph.out_edges(c, data=True):
                        out_traffic += eattr["comm"]
                    SCG.nodes[idx_pt]["switching_activity"]["c_{}".format(c)] = out_traffic / bw

        # check router port num constraint
        flg_fail = False
        for (_, nattr) in SCG.nodes(data=True):
            if nattr["switch_size_in"] > max_port or nattr["switch_size_out"] > max_port:
                flg_fail = True
                break
        if flg_fail:
            continue

        # 4. Check for bandwidth constraint violations.
        comm_matrix = np.zeros(shape=(len(partitions), len(partitions)), dtype=int)
        for i_f, j_f, attr in task_graph.edges(data=True):
            f = attr["comm"]
            if f > 0:
                i_pt = [idx_pt for idx_pt, pt in enumerate(partitions) if i_f in pt]  # find the partition of src i
                j_pt = [idx_pt for idx_pt, pt in enumerate(partitions) if j_f in pt]  # find the partition of dst j
                assert len(i_pt) == 1 and len(j_pt) == 1
                i_pt, j_pt = i_pt[0], j_pt[0]
                if i_pt != j_pt:
                    comm_matrix[i_pt][j_pt] += f
        if (comm_matrix > bw).any():
            continue

        # 5. Build Switch Cost Graph (SCG). The SCG is afully connected graph

        # 6. Build Prohibited Turn Set (PTS) for SCG to avoid deadlocks
        TPS = get_TPS(SCG.to_undirected(as_view=True))

        # 8. Find paths for flows across the switches using function PATH COMPUTE(i, SCG, rho, PTS, theta)
        rtp_SCG = path_compute(pa_db=pa_db, i=i, max_port=max_port, bw=bw, SCG=SCG, TPS=TPS, comm_matrix=comm_matrix)
        if rtp_SCG is None:
            continue
        else:
            assert check_deadlock(rtp_SCG)
            topo_graph = nx.DiGraph()
            for idx_pt, pt in enumerate(partitions):
                for c in pt:
                    if task_graph.in_degree(c):
                        topo_graph.add_edge(V + idx_pt, c)
                    if task_graph.out_degree(c):
                        topo_graph.add_edge(c, V + idx_pt)

            for (s_node, d_node), p_r in rtp_SCG.items():
                p = [s_node] + p_r + [d_node]
                for i_node, j_node in zip(p, p[1:]):
                    topo_graph.add_edge(i_node + V, j_node + V)  # this is important

            rtp = {}
            for (s_node, d_node) in task_graph.edges():
                s_r = [idx_pt for idx_pt, pt in enumerate(partitions) if s_node in pt][0]
                d_r = [idx_pt for idx_pt, pt in enumerate(partitions) if d_node in pt][0]
                if s_r == d_r:
                    rtp[(s_node, d_node)] = [s_r + V]
                else:
                    rtp[(s_node, d_node)] = [s_r + V] + [r + V for r in rtp_SCG[(s_r, d_r)]] + [d_r + V]
            return topo_graph, rtp
