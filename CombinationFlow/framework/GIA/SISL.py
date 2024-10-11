import networkx as nx
from .. import NoC
import copy
import os
import numpy as np
import math
from .ActiveGIA import get_p_reg_128


def mark_topo(topo_graph, cnum, W_intp):
    """
        set the coordinate/input port/output port of nodes in topology graph, some routers will be removed/power down
    """
    for n in topo_graph:
        if n >= cnum:
            topo_graph.nodes[n]["x_coord"] = (n - cnum) % W_intp
            topo_graph.nodes[n]["y_coord"] = (n - cnum) // W_intp
            topo_graph.nodes[n]["in_port"] = topo_graph.in_degree(n)
            topo_graph.nodes[n]["out_port"] = topo_graph.out_degree(n)


def remove_unused(task_graph, topo_graph, rtp):
    """
        Remove the unused routers whose leakage power will influence results
    """
    topo_graph = copy.deepcopy(topo_graph)
    rtp = copy.deepcopy(rtp)
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
    return topo_graph, rtp


def gen_SISL_mesh(csys, placement):
    """
        W, H: size of interposer, unit is tile
        routers in interposer:
            ^ y
            |
            |4:(0,2) 5:(1, 2)
            |2:(0,1) 3:(1, 1)
            |0:(0,0) 1:(1, 0)
            |————————————————> x
    """
    cnum = len(csys.task_graph)
    topo_graph_u = nx.Graph()
    for (x_u, x_v) in zip(range(0, csys.W), range(1, csys.W)):
        for y in range(csys.H):
            topo_graph_u.add_edge(x_u + y * csys.W, x_v + y * csys.W)

    for (y_u, y_v) in zip(range(0, csys.H), range(1, csys.H)):
        for x in range(csys.W):
            topo_graph_u.add_edge(x + y_u * csys.W, x + y_v * csys.W)

    node_mapping = dict(zip(topo_graph_u.nodes(), map(lambda n: cnum + n, topo_graph_u.nodes())))
    topo_graph_u = nx.relabel_nodes(topo_graph_u, node_mapping)  # 0 ~ (cnum - 1) are cores
    topo_graph_u.add_nodes_from(range(cnum))  # add NIs

    for n in csys.task_graph:  # connect NIs in chiplets to interposer routers
        x_n, y_n = csys.get_xy_pin(n, placement)
        rtr = x_n + y_n * csys.W + cnum
        assert rtr < len(topo_graph_u)
        topo_graph_u.add_edge(n, rtr)
    topo_graph = topo_graph_u.to_directed()
    mark_topo(topo_graph=topo_graph, cnum=cnum, W_intp=csys.W)  # mark the original position of topologies

    # generate routing path using x y routing
    # rtp: routing path list dict; rtp[(src node, dst node)] = [rtr0, rtr1, ...]
    rtp = {}
    for (src, dst) in csys.task_graph.edges():
        rtr_src = list(topo_graph.neighbors(src))
        assert len(rtr_src) == 1
        rtr_src = rtr_src[0]
        rtr_dst = list(topo_graph.neighbors(dst))
        assert len(rtr_dst) == 1
        rtr_dst = rtr_dst[0]
        x_rtr_src, y_rtr_src = (rtr_src - cnum) % csys.W, (rtr_src - cnum) // csys.W
        x_rtr_dst, y_rtr_dst = (rtr_dst - cnum) % csys.W, (rtr_dst - cnum) // csys.W

        rtp_this = []
        x_rtr_cur = x_rtr_src
        y_rtr_cur = y_rtr_src
        rtp_this.append(x_rtr_cur + y_rtr_cur * csys.W + cnum)
        while True:
            if x_rtr_cur > x_rtr_dst:
                x_rtr_cur -= 1
            elif x_rtr_cur < x_rtr_dst:
                x_rtr_cur += 1
            elif y_rtr_cur > y_rtr_dst:
                y_rtr_cur -= 1
            elif y_rtr_cur < y_rtr_dst:
                y_rtr_cur += 1
            else:
                break
            rtp_this.append(x_rtr_cur + y_rtr_cur * csys.W + cnum)
        rtp[(src, dst)] = rtp_this
    topo_graph, rtp = remove_unused(csys.task_graph, topo_graph, rtp)
    return topo_graph, rtp


def gen_SISL_ft(csys, placement, bw):
    """
         generate Folded Torus topology interposer
        W, H: size of interposer, unit is tile
        routers in interposer:
            ^ y
            |
            |4:(0,2) 5:(1, 2)
            |2:(0,1) 3:(1, 1)
            |0:(0,0) 1:(1, 0)
            |————————————————> x
    """
    cnum = len(csys.task_graph)
    topo_graph_u = nx.Graph()
    for y in range(csys.H):
        for x in range(csys.W):
            if x + 2 <= csys.W - 1:
                topo_graph_u.add_edge(x + y * csys.W, (x + 2) + y * csys.W)
            if y + 2 <= csys.H - 1:
                topo_graph_u.add_edge(x + y * csys.W, x + (y + 2) * csys.W)

            if x == 0 or x == csys.W - 2:
                topo_graph_u.add_edge(x + y * csys.W, (x + 1) + y * csys.W)
            if y == 0 or y == csys.H - 2:
                topo_graph_u.add_edge(x + y * csys.W, x + (y + 1) * csys.W)

    node_mapping = dict(zip(topo_graph_u.nodes(), map(lambda n: cnum + n, topo_graph_u.nodes())))
    topo_graph_u = nx.relabel_nodes(topo_graph_u, node_mapping)  # 0 ~ (cnum - 1) are cores
    topo_graph_u.add_nodes_from(range(cnum))  # add NIs

    for n in csys.task_graph:  # connect NIs in chiplets to interposer routers
        x_n, y_n = csys.get_xy_pin(n, placement)
        rtr = x_n + y_n * csys.W + cnum
        assert rtr < len(topo_graph_u)
        topo_graph_u.add_edge(n, rtr)

    TPS = NoC.get_TPS(topo_graph_u)
    topo_graph = topo_graph_u.to_directed()
    mark_topo(topo_graph=topo_graph, cnum=cnum, W_intp=csys.W)
    rtp = NoC.routing_base(task_graph=csys.task_graph, topo_graph=topo_graph, TPS=TPS, bw=bw, weight_func=None)
    topo_graph, rtp = remove_unused(csys.task_graph, topo_graph, rtp)
    return topo_graph, rtp


def gen_SISL_bd(csys, placement, bw):
    """
        generate Butter Donut topology interposer
        W, H: size of interposer, unit is tile
        routers in interposer:
            ^ y
            |
            |4:(0,2) 5:(1, 2)
            |2:(0,1) 3:(1, 1)
            |0:(0,0) 1:(1, 0)
            |————————————————> x
    """
    cnum = len(csys.task_graph)
    bd_raw = nx.Graph()  # raw butter donut and need to be clipped
    # generate butter donut and then clip
    stage = math.ceil(csys.W / 2)  # half of total stages
    rtr_stage = min(2**(stage - 1), csys.H)  # router per stage
    for s in range(stage - 1):  # generate Double Butterfly firstly
        for r in range(rtr_stage):
            if (r // (2**s)) % 2 == 0:
                bd_raw.add_edge((r, s), (r + 2**s, s + 1))  # r: row, s: column
            else:
                bd_raw.add_edge((r, s), (r - 2**s, s + 1))
    for s in range(stage - 1):
        for r in range(rtr_stage):
            if (r // (2**s)) % 2 == 0:
                bd_raw.add_edge((r, 2 * stage - 1 - s), (r + 2**s, 2 * stage - 1 - s - 1))
            else:
                bd_raw.add_edge((r, 2 * stage - 1 - s), (r - 2**s, 2 * stage - 1 - s - 1))
    s = stage - 1
    for r in range(rtr_stage):
        if r % 2 == 0:
            bd_raw.add_edge((r, s), (r + 1, s + 1))
        else:
            bd_raw.add_edge((r, s), (r - 1, s + 1))
    for s in range(2 * stage):  # add horizontal links
        for r in range(rtr_stage):
            if s == 0 or s == 2 * stage - 2:
                bd_raw.add_edge((r, s), (r, s + 1))
            if s + 2 <= 2 * stage - 1:
                bd_raw.add_edge((r, s), (r, s + 2))

    # print(bd_raw.edges(), nx.is_connected(bd_raw))
    bd_raw = bd_raw.subgraph([(i, j) for i in range(csys.H) for j in range(csys.W)])
    node_mapping = dict(zip(bd_raw.nodes(), map(lambda ij: ij[0] * csys.W + ij[1], bd_raw.nodes())))
    topo_graph_u = nx.relabel_nodes(bd_raw, node_mapping)
    node_mapping = dict(zip(topo_graph_u.nodes(), map(lambda n: cnum + n, topo_graph_u.nodes())))
    topo_graph_u = nx.relabel_nodes(topo_graph_u, node_mapping)  # 0 ~ (cnum - 1) are cores
    topo_graph_u.add_nodes_from(range(cnum))  # add NIs

    for n in csys.task_graph:  # connect NIs in chiplets to interposer routers
        x_n, y_n = csys.get_xy_pin(n, placement)
        rtr = x_n + y_n * csys.W + cnum
        assert rtr < len(topo_graph_u)
        topo_graph_u.add_edge(n, rtr)

    TPS = NoC.get_TPS(topo_graph_u)
    topo_graph = topo_graph_u.to_directed()
    mark_topo(topo_graph=topo_graph, cnum=cnum, W_intp=csys.W)
    rtp = NoC.routing_base(task_graph=csys.task_graph, topo_graph=topo_graph, TPS=TPS, bw=bw, weight_func=None)
    topo_graph, rtp = remove_unused(csys.task_graph, topo_graph, rtp)
    return topo_graph, rtp


def get_power(dir_dsent, task_graph, topo_graph, rtp, bw, freq, tile_size, path_router_db, path_wire_db):
    """
        Get power of mesh/ft/bd topology.
    """
    assert bw == 128  # Gbit/s, router/wire db is for 128 bit
    assert freq == 1e9
    assert tile_size == 1e-3
    router_db = NoC.load_db_router(path_pa_db=path_router_db)
    wire_db = NoC.load_db_wire(path_p_db=path_wire_db)
    topo_graph = copy.deepcopy(topo_graph)
    nx.set_edge_attributes(topo_graph, 0, name="acc_comm")
    for (src, dst), path_ in rtp.items():
        path = [src, *path_, dst]
        for u, v in zip(path, path[1:]):
            topo_graph[u][v]["acc_comm"] += task_graph[src][dst]["comm"]
    # get power of routers
    power_router = 0
    for n in range(len(task_graph), len(topo_graph)):  # traverse all routers
        in_port = topo_graph.nodes[n]["in_port"]
        out_port = topo_graph.nodes[n]["out_port"]
        load = 0
        for _, __, e_attr in topo_graph.in_edges(n, data=True):
            load += min(e_attr["acc_comm"], bw) / bw
        load /= in_port
        pa_router = NoC.eval_PA_router(pa_db=router_db, in_port=in_port, out_port=out_port, load=load)
        power_router += pa_router["total_power"]
    # print("router power: ", power_router)

    # get power of wires
    power_wire = 0
    cnum = len(task_graph)
    for u, v, e_attr in topo_graph.edges(data=True):
        if u < cnum or v < cnum:  # is NI
            continue
        x_u = topo_graph.nodes[u]["x_coord"]
        y_u = topo_graph.nodes[u]["y_coord"]
        x_v = topo_graph.nodes[v]["x_coord"]
        y_v = topo_graph.nodes[v]["y_coord"]
        length = (abs(x_u - x_v) + abs(y_u - y_v)) * tile_size  # get length of wire
        if length <= 0.02:
            p_wire = NoC.eval_P_wire(p_db=wire_db, load=min(e_attr["acc_comm"] / bw, 1), length=length,
                                     delay=1 / freq)  # wire is short, no need for signal registering
        else:
            assert length < 0.04
            p_wire_0 = NoC.eval_wire_dsent(dir_dsent=dir_dsent,
                                           load=min(e_attr["acc_comm"] / bw, 1),
                                           cfg={
                                               "process": 45,
                                               "freq": freq,
                                               "width": (bw * 1e9) // 1e9,
                                               "length": 0.02,
                                               "delay": 1 / freq
                                           },
                                           clean=True)
            p_wire_1 = NoC.eval_wire_dsent(dir_dsent=dir_dsent,
                                           load=min(e_attr["acc_comm"] / bw, 1),
                                           cfg={
                                               "process": 45,
                                               "freq": freq,
                                               "width": (bw * 1e9) // 1e9,
                                               "length": length - 0.02,
                                               "delay": 1 / freq
                                           },
                                           clean=True)
            p_wire = {
                "dynamic": p_wire_0["dynamic"] + p_wire_1["dynamic"],
                "leakage": p_wire_0["leakage"] + p_wire_1["leakage"]
            }

        power_wire += p_wire["dynamic"] + p_wire["leakage"]
    # print("wire power: ", power_wire)

    # get power of esd and ubump
    power_esd = 0
    power_ubump = 0
    for u, v, e_attr in topo_graph.edges(data=True):
        if u < cnum or v < cnum:
            power_ubump += 0.64 * 1e-6 * min(e_attr["acc_comm"], bw) * (bw * 1e9 / freq)
            power_esd += 200 * 1e-15 * (1 * 1) * freq * (1 / 4) * min(e_attr["acc_comm"] / bw, 1) * (
                bw * 1e9 / freq)  # 50% probability of turn; once per cycle

    # print("esd power: ", power_esd)
    # print("ubump power:", power_ubump)
    return {"power": power_router + power_wire, "power_eu": power_router + power_wire + power_esd + power_ubump}


def SISL_app(dir_dsent, csys, placement, topo_graph, rtp, avg_lat_packet, bw, freq, tile_size, intp_type):
    """
        max_hop_cycle: different for idea active and passive interposer
    """
    assert bw == 128
    assert freq == 1e9
    assert tile_size == 1e-3

    cnum = len(csys.task_graph)
    for n in range(len(topo_graph)):
        if n < cnum:
            topo_graph.nodes[n]["x_coord"], topo_graph.nodes[n]["y_coord"] = csys.get_xy_pin(n, placement)
        else:
            neigh_NI = [n_ for n_ in set(nx.all_neighbors(topo_graph, n)) if n_ < cnum]  #  find all connected NIs
            assert len(neigh_NI) > 0  # can not handle router only connected to routers
            xy_neigh_NI = [csys.get_xy_pin(ni, placement) for ni in neigh_NI]
            x_neigh_NI, y_neigh_NI = list(zip(*xy_neigh_NI))
            x_mid = np.median(x_neigh_NI)
            y_mid = np.median(y_neigh_NI)
            topo_graph.nodes[n]["x_coord"] = x_mid
            topo_graph.nodes[n]["y_coord"] = y_mid

    nx.set_edge_attributes(topo_graph, 0, name="acc_comm")
    for (src, dst), path_ in rtp.items():
        path = [src, *path_, dst]
        for u, v in zip(path, path[1:]):
            topo_graph[u][v]["acc_comm"] += csys.task_graph[src][dst]["comm"]

    router_db = NoC.load_db_router(path_pa_db=os.path.join(dir_dsent, "router_pa_db.json"))
    # get router power
    power_router = 0
    for n in range(len(csys.task_graph), len(topo_graph)):  # traverse all routers
        in_port = topo_graph.in_degree(n)
        out_port = topo_graph.out_degree(n)
        load = 0
        for _, __, e_attr in topo_graph.in_edges(n, data=True):
            load += min(e_attr["acc_comm"], bw) / bw
        load /= in_port
        pa_router = NoC.eval_PA_router(pa_db=router_db, in_port=in_port, out_port=out_port, load=load)
        power_router += pa_router["total_power"]
    # print("router power: ", power_router)

    power_wire = 0
    load_resurface = []
    for u, v, e_attr in topo_graph.edges(data=True):
        x_u = topo_graph.nodes[u]["x_coord"]
        y_u = topo_graph.nodes[u]["y_coord"]
        x_v = topo_graph.nodes[v]["x_coord"]
        y_v = topo_graph.nodes[v]["y_coord"]
        length = (abs(x_u - x_v) + abs(y_u - y_v)) * tile_size  # physical length, unit is meter
        load = min(e_attr["acc_comm"] / bw, 1)
        if intp_type == "active":
            max_len = 20 * 1e-3  # stow
            num_reg = int(length / max_len)
            p_wire_unit = NoC.eval_wire_dsent(dir_dsent=dir_dsent,
                                              load=load,
                                              cfg={
                                                  "process": 45,
                                                  "freq": freq,
                                                  "width": (bw * 1e9) // 1e9,
                                                  "length": max_len,
                                                  "delay": 1 / freq
                                              },
                                              clean=True)
            p_wire_resi = NoC.eval_wire_dsent(dir_dsent=dir_dsent,
                                              load=load,
                                              cfg={
                                                  "process": 45,
                                                  "freq": freq,
                                                  "width": (bw * 1e9) // 1e9,
                                                  "length": length - max_len * num_reg,
                                                  "delay": 1 / freq
                                              },
                                              clean=True)
            p_wire = {
                "dynamic": p_wire_unit["dynamic"] * num_reg + p_wire_resi["dynamic"],
                "leakage": p_wire_unit["leakage"] * num_reg + p_wire_resi["leakage"]
            }
        elif intp_type == "passive":
            max_len = 7 * 1e-3  # simulation with Intel 45nm
            num_reg = int(length / max_len)
            p_wire_unit = NoC.eval_wire_dsent(dir_dsent=dir_dsent,
                                              load=load,
                                              cfg={
                                                  "process": 45,
                                                  "freq": freq,
                                                  "width": (bw * 1e9) // 1e9,
                                                  "length": max_len,
                                                  "delay": 1 / freq
                                              },
                                              clean=True)
            p_wire_resi = NoC.eval_wire_dsent(dir_dsent=dir_dsent,
                                              load=load,
                                              cfg={
                                                  "process": 45,
                                                  "freq": freq,
                                                  "width": (bw * 1e9) // 1e9,
                                                  "length": length - max_len * num_reg,
                                                  "delay": 1 / freq
                                              },
                                              clean=True)
            p_wire = {
                "dynamic": p_wire_unit["dynamic"] * num_reg + p_wire_resi["dynamic"],
                "leakage": p_wire_unit["leakage"] * num_reg + p_wire_resi["leakage"]
            }
            load_resurface += [load] * num_reg
        else:
            assert False
        topo_graph[u][v]["num_reg"] = num_reg
        power_wire += p_wire["dynamic"] + p_wire["leakage"] + num_reg * get_p_reg_128(load=load)

    # get power of esd and ubump
    power_esd = 0
    power_ubump = 0

    for u, v, e_attr in topo_graph.edges(data=True):
        if intp_type == "active":
            if u < cnum or v < cnum:  # connections of routers in interposer do not need ubumps
                power_ubump += 0.64 * 1e-6 * min(e_attr["acc_comm"], bw) * (bw * 1e9 / freq)
                power_esd += 200 * 1e-15 * (1 * 1) * freq * (1 / 4) * min(e_attr["acc_comm"] / bw, 1) * (
                    bw * 1e9 / freq)  # 1/4 = 1/2 * 1/2, 50% probability of turn; once per cycle
        else:
            power_ubump += 0.64 * 1e-6 * min(e_attr["acc_comm"], bw) * (bw * 1e9 / freq) * 2
            power_esd += 200 * 1e-15 * (1 * 1) * freq * (1 / 4) * min(e_attr["acc_comm"] / bw, 1) * (bw * 1e9 / freq) * 2

    for l in load_resurface:
        power_ubump += 0.64 * 1e-6 * (l * bw) * (bw * 1e9 / freq) * 2
        power_esd += 200 * 1e-15 * (1 * 1) * freq * (1 / 4) * l * (bw * 1e9 / freq) * 2

    lat_penal = 0
    total_traffic = sum([e_attr["comm"] for _, __, e_attr in csys.task_graph.edges(data=True)])
    for (src, dst), path in rtp.items():
        lat_penal += (-1) * len(
            path) * csys.task_graph[src][dst]["comm"] / total_traffic  # minus 1 for redundant 1 cycle crossbar traversal delay
        for u, v in zip(path, path[1:]):
            lat_penal += (csys.task_graph[src][dst]["comm"] * topo_graph[u][v]["num_reg"]) / total_traffic
    PPP = {}
    PPP["power"] = power_router + power_wire
    PPP["power_eu"] = power_router + power_wire + power_esd + power_ubump
    PPP["perf"] = avg_lat_packet + lat_penal
    PPP["avg_lat_packet_orig"] = avg_lat_packet
    if intp_type == "active":
        PPP["pcost"] = len([(u, v) for (u, v) in topo_graph.edges() if u < cnum or v < cnum])
    else:
        PPP["pcost"] = len(load_resurface) * 2 + len(topo_graph.edges()) * 2
    return PPP


def SISL(dir_dsent,
         dir_booksim,
         topo_type,
         csys,
         placement,
         tile_size,
         cfg_booksim,
         topo_graph=None,
         rtp=None,
         avg_lat_packet=None,
         intp_type=None):
    """
        csys: ChipletSys instance
        topo_type: "mesh/ft/bd/app" for mesh/folded torus/butter donut
        cfg_booksim: cfg of eval_PPA_booksim
        topo_graph/rtp/avg_lat_packet/intp_type: only for application-specific SISL

        return: dict of power, latency
    """
    bw = 128  # Gbit/s
    freq = 1e9  # Hz
    assert tile_size == 1e-3
    if topo_type == "mesh":
        topo_graph, rtp = gen_SISL_mesh(csys=csys, placement=placement)
    elif topo_type == "ft":
        topo_graph, rtp = gen_SISL_ft(csys=csys, placement=placement, bw=bw)
    elif topo_type == "bd":
        topo_graph, rtp = gen_SISL_bd(csys=csys, placement=placement, bw=bw)
    elif topo_type == "app":
        PPP = SISL_app(dir_dsent=dir_dsent,
                       csys=csys,
                       placement=placement,
                       topo_graph=topo_graph,
                       rtp=rtp,
                       avg_lat_packet=avg_lat_packet,
                       bw=bw,
                       freq=freq,
                       tile_size=tile_size,
                       intp_type=intp_type)
        return PPP

    if topo_type != "app":
        power = get_power(dir_dsent=dir_dsent,
                          task_graph=csys.task_graph,
                          topo_graph=topo_graph,
                          rtp=rtp,
                          bw=bw,
                          freq=freq,
                          tile_size=tile_size,
                          path_router_db=os.path.join(dir_dsent, "router_pa_db.json"),
                          path_wire_db=os.path.join(dir_dsent, "wire_p_db.json"))
        try:
            PPA_booksim = NoC.eval_PPA_booksim(dir_booksim=dir_booksim,
                                               task_graph=csys.task_graph,
                                               topo_graph=topo_graph,
                                               rtp=rtp,
                                               cfg=cfg_booksim,
                                               clean=True)
        except Exception as e:
            print(e)
            PPA_booksim = {"avg_lat_packet": 1e9}
        lat_penal = 0
        total_traffic = sum([e_attr["comm"] for _, __, e_attr in csys.task_graph.edges(data=True)])
        for (src, dst), path in rtp.items():
            lat_penal += (-1) * len(path) * csys.task_graph[src][dst][
                "comm"] / total_traffic  # minus 1 for redundant 1 cycle crossbar traversal delay

        PPP = {}
        PPP["power"] = power["power"]
        PPP["power_eu"] = power["power_eu"]
        PPP["perf"] = PPA_booksim["avg_lat_packet"] + lat_penal
        PPP["avg_lat_packet_orig"] = PPA_booksim["avg_lat_packet"]
        PPP["pcost"] = len([(u, v) for (u, v) in topo_graph.edges() if u < len(csys.task_graph) or v < len(csys.task_graph)])
        return PPP