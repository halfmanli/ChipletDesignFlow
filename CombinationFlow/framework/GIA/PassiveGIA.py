from .PassiveLayout import gen_layout_passive, get_DIR, is_covered_cpl
from .ActiveGIA import get_p_mux_8_1, get_p_reg_128
from .. import NoC
import os
import networkx as nx


def PassiveGIA(dir_dsent, csys, placement, topo_graph, rtp, tile_size, bw, freq):
    """
        Return is a dict containing "power" and "lat_penal"
    """
    assert bw == 128  # Gbit/s, router/wire db is for 128 bit
    assert freq == 1e9
    cq_setup = (33.9 + 34)  # unit is ps
    delay_table = [None, 136, 209, 305, 423, 567, 732, 921, 1e8]  # unit is ps
    delay_xbar = 30  # unit is ps

    flg_suc, xy_pr, paths = gen_layout_passive(csys=csys,
                                               topo_graph=topo_graph,
                                               rtp=rtp,
                                               placement=placement,
                                               faulted_links=None,
                                               base_cost=1,
                                               bend_cost=1,
                                               y_cost=0,
                                               max_retry=90)
    if flg_suc is False:  # failed to generate mlayout
        return None

    pnum = len(csys.task_graph)  # number of NI
    router_db = NoC.load_db_router(path_pa_db=os.path.join(dir_dsent, "router_pa_db.json"))
    wire_db = NoC.load_db_wire(path_p_db=os.path.join(dir_dsent, "wire_p_db.json"))
    topo_graph = topo_graph.copy()
    nx.set_edge_attributes(topo_graph, 0, name="acc_comm")
    for (src, dst), path_ in rtp.items():
        path = [src, *path_, dst]
        for u, v in zip(path, path[1:]):
            topo_graph[u][v]["acc_comm"] += csys.task_graph[src][dst]["comm"]

    # get power of routers
    power_router = 0
    for n in range(len(csys.task_graph), len(topo_graph)):  # traverse all routers
        in_port = topo_graph.in_degree(n)
        out_port = topo_graph.out_degree(n)
        assert in_port <= 4 and out_port <= 4
        load = 0
        for _, __, e_attr in topo_graph.in_edges(n, data=True):
            load += e_attr["acc_comm"] / bw
        load /= in_port
        pa_router = NoC.eval_PA_router(pa_db=router_db, in_port=in_port, out_port=out_port, load=load)
        power_router += pa_router["total_power"] - (pa_router["xbar_dynamic"] +
                                                    pa_router["xbar_leakage"]) + get_p_mux_8_1(load) * out_port

    total_traffic = sum([e_attr["comm"] for _, __, e_attr in csys.task_graph.edges(data=True)])

    lat_penal = 0
    for (src, dst), path in rtp.items():  # minus 1 for redundant 1 cycle crossbar traversal delay
        lat_penal += (-1) * len(path) * csys.task_graph[src][dst]["comm"] / total_traffic

    xy_resurface = []  # (x, y) of the tile for resurfacing/going back & going down
    xy_bgcpl = []  # resurface at position having no chiplet
    power_wire = 0  # get power of wires = metal wire + registers + crossbars
    for (s, d), p in paths.items():
        load = topo_graph[s][d][
            "acc_comm"] / bw  # this is correct because one physical wire will be used by only one topological link
        assert load <= 1
        num_mux = 0  # number of used mux for signal turn in this path
        num_reg = 0  # number of used reg for signal registering in this path
        dist_cur = 0  # distance of going straight and without buffering
        delay_rcd = []  # delay record, element is (wire delay or xbar delay, xy of wire sink or xbar position)
        p_wire = 0  # pure wire power(without xbar/register) of this path
        for idx_pp, pp in enumerate(p):  # pp: ((x_u, y_u), (x_v, y_v), channel id)
            turn = False
            if idx_pp != 0:
                u_m, v_m, c_m = p[idx_pp - 1]
                u_n, v_n, c_n = p[idx_pp]
                if (get_DIR(u_m, v_m) != get_DIR(u_n, v_n) or c_m != c_n):
                    turn = True
            if turn:
                delay_rcd.append((delay_table[dist_cur], pp[0]))
                p_wire_raw = NoC.eval_P_wire(p_db=wire_db,
                                             load=load,
                                             length=dist_cur * tile_size,
                                             delay=delay_table[dist_cur] * 1e-12)
                p_wire += p_wire_raw["dynamic"] + p_wire_raw["leakage"]
                delay_rcd.append((delay_xbar, pp[0]))
                dist_cur = 0
                xy_resurface.append((pp[0], load))
                num_mux += 1
            else:
                if delay_table[dist_cur + 1] + cq_setup > (1 / freq) * 1e12:  # static timing constraint
                    delay_rcd.append((delay_table[dist_cur], pp[0]))
                    p_wire_raw = NoC.eval_P_wire(p_db=wire_db,
                                                 load=load,
                                                 length=dist_cur * tile_size,
                                                 delay=delay_table[dist_cur] * 1e-12)
                    p_wire += p_wire_raw["dynamic"] + p_wire_raw["leakage"]
                    xy_resurface.append((pp[0], load))
                    dist_cur = 0
            dist_cur += 1

        delay_acc = 0
        xy_last = None  # xy of last/previous record
        for r in delay_rcd:
            if delay_acc + r[0] + cq_setup >= (1 / freq) * 1e12:
                if xy_last is not None:
                    xy_resurface.append((xy_last, load))
                delay_acc = r[0]
                num_reg += 1
            else:
                delay_acc += r[0]
            xy_last = r[1]
        p_mux = get_p_mux_8_1(load) * num_mux
        p_reg = get_p_reg_128(load) * num_reg

        power_wire += p_wire + p_mux + p_reg
        lat_penal += num_reg * topo_graph[s][d]["acc_comm"] / total_traffic

    # get power of esd and ubump
    power_esd = 0
    power_ubump = 0
    num_ubump = 0
    for u, v, e_attr in topo_graph.edges(data=True):
        if xy_pr[u] != xy_pr[v]:  # not in same tile
            power_ubump += 0.64 * 1e-6 * min(e_attr["acc_comm"],
                                             bw) * (bw * 1e9 / freq) * 2  # * 2 for router not placed in interposer
            power_esd += 200 * 1e-15 * (1 * 1) * freq * (1 / 4) * min(e_attr["acc_comm"] / bw, 1) * (bw * 1e9 / freq) * 2
            num_ubump += 2

    xy_resurface = set(xy_resurface)
    for _, l in xy_resurface:
        power_ubump += 0.64 * 1e-6 * (l * bw) * (bw * 1e9 / freq) * 2
        power_esd += 200 * 1e-15 * (1 * 1) * freq * (1 / 4) * l * (bw * 1e9 / freq) * 2
        num_ubump += 2

    # get bridge chiplet number
    for pr, xy in xy_pr.items():
        if pr >= pnum and not is_covered_cpl(csys=csys, placement=placement, x=xy[0], y=xy[1]):
            xy_bgcpl.append(xy_pr[pr])

    for xy, _ in xy_resurface:
        if xy not in xy_bgcpl and not is_covered_cpl(csys=csys, placement=placement, x=xy[0], y=xy[1]):
            xy_bgcpl.append(xy)

    return {
        "power": power_router + power_wire,
        "power_eu": power_router + power_wire + power_esd + power_ubump,
        "power_detail": {
            "power_router": power_router,
            "power_wire": power_wire,
            "power_esd": power_esd,
            "power_ubump": power_ubump
        },
        "pcost": num_ubump,
        "pcost_detail": {
            "num_ubump": num_ubump,
            "num_bgcpl": len(xy_bgcpl)
        },
        "lat_penal": lat_penal,
        "mlayout": (xy_pr, paths)
    }
