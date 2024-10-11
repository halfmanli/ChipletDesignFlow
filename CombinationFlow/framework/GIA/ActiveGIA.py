from .ActiveLayout import gen_layout_active
from .. import NoC
import os
import networkx as nx


def get_p_mux_8_1(load):
    return (2.496158 * 1e-2 / 8 * load) + (5.213689 * 1e-3 / 8)


def get_p_reg_128(load):
    return (2.150397 * 1e-3 * load) + (5.086871 * 1e-4)


def ActiveGIA(dir_dsent, csys, placement, topo_graph, rtp, tile_size, bw, freq):
    """
        Return is a dict containing "power" and "lat_penal"
    """
    assert bw == 128  # Gbit/s, router/wire db is for 128 bit
    assert freq == 1e9
    assert tile_size == 1e-3
    max_hop_cycle = 12  # (50 + 30) * (max_hop_cycle - 1) + 50 + (33.9 + 34) <= 1000

    flg_suc, xy_pr, paths = gen_layout_active(csys=csys,
                                              topo_graph=topo_graph,
                                              rtp=rtp,
                                              placement=placement,
                                              faulted_links=None,
                                              bend_cost=0,
                                              max_retry=90)
    if flg_suc is False:
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

    power_wire = 0  # get power of wires = metal wire + registers + crossbars
    for (s, d), p in paths.items():
        load = topo_graph[s][d][
            "acc_comm"] / bw  # this is correct because one physical wire will be used by only one topological link
        assert load <= 1
        hop = len(p) - 1
        num_reg = (hop - 1) // max_hop_cycle
        p_mux = get_p_mux_8_1(load) * (hop - 1)
        p_reg = get_p_reg_128(load) * num_reg
        p_wire = NoC.eval_P_wire(p_db=wire_db, load=load, length=1 * tile_size, delay=50 * 1e-12)
        power_wire += (p_wire["dynamic"] + p_wire["leakage"]) * hop + p_mux + p_reg
        lat_penal += num_reg * topo_graph[s][d]["acc_comm"] / total_traffic
    # print("router power", power_router, "wire power: ", power_wire)

    # get power of esd and ubump
    power_esd = 0
    power_ubump = 0
    for u, v, e_attr in topo_graph.edges(data=True):
        if u < pnum or v < pnum:
            power_ubump += 0.64 * 1e-6 * min(e_attr["acc_comm"], bw) * (bw * 1e9 / freq
                                                                        )  #  routers are in interposer, no need to * 2
            power_esd += 200 * 1e-15 * (1 * 1) * freq * (1 / 4) * min(e_attr["acc_comm"] / bw, 1) * (
                bw * 1e9 / freq)  # 50% probability of turn; once per cycle

    # print("esd power: ", power_esd)
    # print("ubump power:", power_ubump)
    return {
        "power": power_router + power_wire,
        "power_eu": power_router + power_wire + power_esd + power_ubump,
        "lat_penal": lat_penal,
        "pcost": len([(u, v) for (u, v) in topo_graph.edges() if u < pnum or v < pnum]),
        "mlayout": (xy_pr, paths)
    }
