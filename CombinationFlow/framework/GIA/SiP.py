import subprocess
from .SiPLayout import gen_layout_sip, get_DIR, is_covered_cpl
from .. import NoC
from bidict import bidict
import copy
import networkx as nx
import os
from .ActiveGIA import get_p_mux_8_1, get_p_reg_128


def LCSubLst(a, b):
    m = len(a)
    n = len(b)
    LCSuff = [[0 for k in range(n + 1)] for l in range(m + 1)]
    length = 0
    idx_a, idx_b = -1, -1

    # Following steps to build
    # LCSuff[m+1][n+1] in bottom up fashion
    for i in range(m + 1):
        for j in range(n + 1):
            if (i == 0 or j == 0):
                LCSuff[i][j] = 0
            elif (a[i - 1] == b[j - 1]):
                LCSuff[i][j] = LCSuff[i - 1][j - 1] + 1
                if length < LCSuff[i][j]:
                    length = LCSuff[i][j]
                    idx_a, idx_b = i - 1 - length + 1, j - 1 - length + 1
            else:
                LCSuff[i][j] = 0
    return length, idx_a, idx_b


def rpl_aux(rtp, old, new):
    assert type(old) == tuple and type(new) == tuple and len(old) == 2
    rtp_new = {}
    for (s, d), path in rtp.items():
        path_new = [s] + path + [d]
        while True:
            e = list(zip(path_new, path_new[1:]))
            if old in e:
                idx = e.index(old)
                path_new = path_new[:idx] + list(new) + path_new[idx + 2:]
            else:
                break
        rtp_new[(path_new[0], path_new[-1])] = path_new[1:-1]

    return rtp_new


def fix_contention(csys, topo_graph, rtp, paths, xy_pr):
    topo_graph = copy.deepcopy(topo_graph)
    rtp = copy.deepcopy(rtp)
    paths = copy.deepcopy(paths)
    xy_pr = copy.deepcopy(xy_pr)

    pnum = len(csys.task_graph)
    xy_r = bidict([(n, xy_pr[n]) for n in xy_pr if n >= pnum])  # xy of all routers
    while True:
        flg_conflict = False
        for (s1, d1), p1_ in paths.items():
            p1 = [(p[0], p[1]) for p in p1_]  # remove channel id
            for (s2, d2), p2_ in paths.items():
                if (s1, d1) == (s2, d2):
                    continue

                p2 = [(p[0], p[1]) for p in p2_]
                length, idx_p1, idx_p2 = LCSubLst(p1, p2)
                if length and (s1 >= pnum and d1 >= pnum and s2 >= pnum and d2 >= pnum):
                    xy_rtr_in, xy_rtr_out = p1[idx_p1][0], p1[idx_p1 + length - 1][1]

                    if xy_rtr_in not in xy_r.inverse:
                        rtr_in = len(topo_graph)
                        topo_graph.add_node(rtr_in)  # insert new router
                        xy_r[rtr_in] = xy_rtr_in
                    else:
                        rtr_in = xy_r.inverse[xy_rtr_in]
                    if xy_rtr_out not in xy_r.inverse:
                        rtr_out = len(topo_graph)
                        topo_graph.add_node(rtr_out)
                        xy_r[rtr_out] = xy_rtr_out
                    else:
                        rtr_out = xy_r.inverse[xy_rtr_out]
                    flg_conflict = True

                    paths_new = dict([(k, v) for (k, v) in paths.items() if k != (s1, d1) and k != (s2, d2)])
                    topo_graph.remove_edge(s1, d1)
                    topo_graph.remove_edge(s2, d2)

                    if rtr_in != s1:
                        topo_graph.add_edge(s1, rtr_in)
                        paths_new[(s1, rtr_in)] = paths[(s1, d1)][:idx_p1]

                    if rtr_in != s2:
                        topo_graph.add_edge(s2, rtr_in)
                        paths_new[(s2, rtr_in)] = paths[(s2, d2)][:idx_p2]
                    assert rtr_in != rtr_out
                    topo_graph.add_edge(rtr_in, rtr_out)
                    paths_new[(rtr_in, rtr_out)] = paths[(s1, d1)][idx_p1:idx_p1 + length]

                    if rtr_out != d1:
                        topo_graph.add_edge(rtr_out, d1)
                        paths_new[(rtr_out, d1)] = paths[(s1, d1)][idx_p1 + length:]

                    if rtr_out != d2:
                        topo_graph.add_edge(rtr_out, d2)
                        paths_new[(rtr_out, d2)] = paths[(s2, d2)][idx_p2 + length:]

                    if rtr_in != s1 and rtr_out != d1:
                        rtp = rpl_aux(rtp=rtp, old=(s1, d1), new=(s1, rtr_in, rtr_out, d1))
                    elif rtr_in != s1 and rtr_out == d1:
                        rtp = rpl_aux(rtp=rtp, old=(s1, d1), new=(s1, rtr_in, d1))
                    elif rtr_in == s1 and rtr_out != d1:
                        rtp = rpl_aux(rtp=rtp, old=(s1, d1), new=(s1, rtr_out, d1))

                    if rtr_in != s2 and rtr_out != d2:
                        rtp = rpl_aux(rtp=rtp, old=(s2, d2), new=(s2, rtr_in, rtr_out, d2))
                    elif rtr_in != s2 and rtr_out == d2:
                        rtp = rpl_aux(rtp=rtp, old=(s2, d2), new=(s2, rtr_in, d2))
                    elif rtr_in == s2 and rtr_out != d2:
                        rtp = rpl_aux(rtp=rtp, old=(s2, d2), new=(s2, rtr_out, d2))

                    break

                elif length and ((s1, d1) != (d2, s2)):  # remove (s1, d1) != (d2, s2)
                    flg_conflict = True
                    xy_rtr_in, xy_rtr_out = p1[idx_p1][0], p1[idx_p1 + length - 1][1]
                    if True:
                        if xy_rtr_in not in xy_r.inverse:
                            rtr_in = len(topo_graph)
                            topo_graph.add_node(rtr_in)  # insert new router
                            xy_r[rtr_in] = xy_rtr_in
                        else:
                            rtr_in = xy_r.inverse[xy_rtr_in]
                        if xy_rtr_out not in xy_r.inverse:
                            rtr_out = len(topo_graph)
                            topo_graph.add_node(rtr_out)
                            xy_r[rtr_out] = xy_rtr_out
                        else:
                            rtr_out = xy_r.inverse[xy_rtr_out]

                        paths_new = dict([(k, v) for (k, v) in paths.items()
                                          if k != (s1, d1) and k != (s2, d2) and k != (d1, s1) and k != (d2, s2)])
                        topo_graph.remove_edge(s1, d1)
                        topo_graph.remove_edge(s2, d2)
                        rev1 = (s1 < pnum or d1 < pnum) and topo_graph.has_edge(d1, s1)  # need to reverse
                        rev2 = (s2 < pnum or d2 < pnum) and topo_graph.has_edge(d2, s2)
                        if rev1:
                            topo_graph.remove_edge(d1, s1)
                        if rev2:
                            topo_graph.remove_edge(d2, s2)

                        if rtr_in != s1:
                            topo_graph.add_edge(s1, rtr_in)
                            if rev1:
                                topo_graph.add_edge(rtr_in, s1)
                                paths_new[(rtr_in, s1)] = [(v, u, c) for (u, v, c) in paths[(s1, d1)][:idx_p1]][::-1]
                            paths_new[(s1, rtr_in)] = paths[(s1, d1)][:idx_p1]

                        if rtr_in != s2:
                            if (s1, d1) != (d2, s2):
                                topo_graph.add_edge(s2, rtr_in)
                                if rev2:
                                    topo_graph.add_edge(rtr_in, s2)
                                    paths_new[(rtr_in, s2)] = [(v, u, c) for (u, v, c) in paths[(s2, d2)][:idx_p2]][::-1]
                                paths_new[(s2, rtr_in)] = paths[(s2, d2)][:idx_p2]

                        # if not (rtr_in == s1 and rtr_out == d1) or not (rtr_in == s2 and rtr_out == d2):
                        topo_graph.add_edge(rtr_in, rtr_out)
                        paths_new[(rtr_in, rtr_out)] = paths[(s1, d1)][idx_p1:idx_p1 + length]
                        if rev1 or rev2:
                            topo_graph.add_edge(rtr_out, rtr_in)

                        if rtr_out != d1:
                            topo_graph.add_edge(rtr_out, d1)
                            if rev1:
                                topo_graph.add_edge(d1, rtr_out)
                                paths_new[(d1, rtr_out)] = [(v, u, c) for (u, v, c) in paths[(s1, d1)][idx_p1 + length:]][::-1]
                            paths_new[(rtr_out, d1)] = paths[(s1, d1)][idx_p1 + length:]

                        if rtr_out != d2:
                            if (s1, d1) != (d2, s2):
                                topo_graph.add_edge(rtr_out, d2)
                                if rev2:
                                    topo_graph.add_edge(d2, rtr_out)
                                    paths_new[(d2, rtr_out)] = [(v, u, c)
                                                                for (u, v, c) in paths[(s2, d2)][idx_p2 + length:]][::-1]
                                paths_new[(rtr_out, d2)] = paths[(s2, d2)][idx_p2 + length:]

                        if rtr_in != s1 and rtr_out != d1:
                            rtp = rpl_aux(rtp=rtp, old=(s1, d1), new=(s1, rtr_in, rtr_out, d1))
                            if rev1:
                                rtp = rpl_aux(rtp=rtp, old=(d1, s1), new=(d1, rtr_out, rtr_in, s1))
                        elif rtr_in != s1 and rtr_out == d1:
                            rtp = rpl_aux(rtp=rtp, old=(s1, d1), new=(s1, rtr_in, d1))
                            if rev1:
                                rtp = rpl_aux(rtp=rtp, old=(d1, s1), new=(d1, rtr_in, s1))
                        elif rtr_in == s1 and rtr_out != d1:
                            rtp = rpl_aux(rtp=rtp, old=(s1, d1), new=(s1, rtr_out, d1))
                            if rev1:
                                rtp = rpl_aux(rtp=rtp, old=(d1, s1), new=(d1, rtr_out, s1))

                        if rtr_in != s2 and rtr_out != d2:
                            rtp = rpl_aux(rtp=rtp, old=(s2, d2), new=(s2, rtr_in, rtr_out, d2))
                            if rev2:
                                rtp = rpl_aux(rtp=rtp, old=(d2, s2), new=(d2, rtr_out, rtr_in, s2))
                        elif rtr_in != s2 and rtr_out == d2:
                            rtp = rpl_aux(rtp=rtp, old=(s2, d2), new=(s2, rtr_in, d2))
                            if rev2:
                                rtp = rpl_aux(rtp=rtp, old=(d2, s2), new=(d2, rtr_in, s2))
                        elif rtr_in == s2 and rtr_out != d2:
                            rtp = rpl_aux(rtp=rtp, old=(s2, d2), new=(s2, rtr_out, d2))
                            if rev2:
                                rtp = rpl_aux(rtp=rtp, old=(d2, s2), new=(d2, rtr_out, s2))
                        break

            if flg_conflict:
                break

        if flg_conflict:
            paths = paths_new
        if not flg_conflict:
            break
    """
    for (src, dst) in csys.task_graph.edges():
        if not (src, dst) in rtp:
            print("Fatal ERROR: (src, dst) in rtp")

    for (src, dst) in rtp:
        if not (src, dst) in csys.task_graph.edges():
            print("Fatal ERROR: (src, dst) in csys.task_graph.edges()")

    for (src, dst), path in rtp.items():
        if not topo_graph.has_edge(src, path[0]):
            print("Fatal ERROR: topo_graph.has_edge(src, path[0])")
        if not topo_graph.has_edge(path[-1], dst):
            print("Fatal ERROR: topo_graph.has_edge(path[-1], dst)")
        for r_m, r_n in zip(path, path[1:]):
            if not topo_graph.has_edge(r_m, r_n):
                print("Fatal ERROR: topo_graph.has_edge(r_m, r_n)")
    """
    xy_pr.update(xy_r)
    return topo_graph, rtp, xy_pr, paths


def SiP(dir_dsent, dir_booksim, csys, placement, topo_graph, rtp, tile_size, bw, freq):
    """
        Return is a dict containing "power" and "lat_penal"
    """
    assert bw == 128  # Gbit/s, router/wire db is for 128 bit
    assert freq == 1e9
    cq_setup = (33.9 + 34)  # unit is ps
    delay_table = [None, 136, 209, 305, 423, 567, 732, 921, 1e8]  # unit is ps
    delay_xbar = 30  # unit is ps

    flg_suc, xy_pr, paths = gen_layout_sip(csys=csys,
                                           topo_graph=topo_graph,
                                           rtp=rtp,
                                           placement=placement,
                                           faulted_links=None,
                                           base_cost=1,
                                           bend_cost=1,
                                           y_cost=1,
                                           max_retry=45)

    if flg_suc is False:
        topo_graph, rtp, xy_pr, paths = fix_contention(csys=csys, topo_graph=topo_graph, rtp=rtp, paths=paths, xy_pr=xy_pr)
    try:
        PPA_booksim = NoC.eval_PPA_booksim(dir_booksim=dir_booksim,
                                           task_graph=csys.task_graph,
                                           topo_graph=topo_graph,
                                           rtp=rtp,
                                           cfg={
                                               "sim_cycle": 10000,
                                               "num_vcs": 4,  # TODO: old value is 128
                                               "vc_buf_size": 4,
                                               "timeout": 90
                                           },
                                           clean=True)
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print("ERROR: booksim running in SiP.py failed:", e)
        raise RuntimeError()
    else:
        avg_lat_packet = PPA_booksim["avg_lat_packet"]

    pnum = len(csys.task_graph)  # number of NI
    router_db = NoC.load_db_router(path_pa_db=os.path.join(dir_dsent, "router_pa_db.json"))
    wire_db = NoC.load_db_wire(path_p_db=os.path.join(dir_dsent, "wire_p_db.json"))
    nx.set_edge_attributes(topo_graph, 0, name="acc_comm")
    for (src, dst), path_ in rtp.items():
        path = [src, *path_, dst]
        for u, v in zip(path, path[1:]):
            topo_graph[u][v]["acc_comm"] += csys.task_graph[src][dst]["comm"]

    # get power of routers
    power_router = 0
    for n in range(pnum, len(topo_graph)):  # traverse all routers
        in_port = topo_graph.in_degree(n)
        out_port = topo_graph.out_degree(n)
        # assert in_port <= 4 and out_port <= 4
        load = 0
        for _, __, e_attr in topo_graph.in_edges(n, data=True):
            load += e_attr["acc_comm"] / bw
        load /= in_port
        load = min(load, 1)
        pa_router = NoC.eval_PA_router(pa_db=router_db, in_port=in_port, out_port=out_port, load=load)
        power_router += pa_router["total_power"] - (pa_router["xbar_dynamic"] +
                                                    pa_router["xbar_leakage"]) + get_p_mux_8_1(load) * out_port

    total_traffic = sum([e_attr["comm"] for _, __, e_attr in csys.task_graph.edges(data=True)])

    lat_penal = 0
    for (src, dst), path in rtp.items():  # minus 1 for redundant 1 cycle crossbar traversal delay
        lat_penal += (-1) * len(path) * csys.task_graph[src][dst]["comm"] / total_traffic

    xy_resurface_reg = []  # (x, y) of the tile for resurfacing/going back & going down for signal turning
    xy_resurface_turn = []  # go back to bridge chiplet or go down to interposer for signal turn
    xy_bgcpl = []  # resurface at position having no chiplet
    power_wire = 0  # get power of wires = metal wire + registers + crossbars
    for (s, d), p in paths.items():
        load = topo_graph[s][d][
            "acc_comm"] / bw  # this is correct because one physical wire will be used by only one topological link
        load = min(load, 1)
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
                xy_resurface_turn.append((pp[0], load))
                num_mux += 1
            else:
                if delay_table[dist_cur + 1] + cq_setup > (1 / freq) * 1e12:  # static timing constraint
                    delay_rcd.append((delay_table[dist_cur], pp[0]))
                    p_wire_raw = NoC.eval_P_wire(p_db=wire_db,
                                                 load=load,
                                                 length=dist_cur * tile_size,
                                                 delay=delay_table[dist_cur] * 1e-12)
                    p_wire += p_wire_raw["dynamic"] + p_wire_raw["leakage"]
                    xy_resurface_reg.append((pp[0], load))
                    dist_cur = 0
            dist_cur += 1

        delay_acc = 0
        xy_last = None  # xy of last/previous record
        for r in delay_rcd:
            if delay_acc + r[0] + cq_setup >= (1 / freq) * 1e12:
                if xy_last is not None:
                    xy_resurface_reg.append((xy_last, load))
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
            power_ubump += 0.64 * 1e-6 * min(e_attr["acc_comm"], bw) * (bw * 1e9 / freq) * 2
            power_esd += 200 * 1e-15 * (1 * 1) * freq * (1 / 4) * min(
                e_attr["acc_comm"] / bw, 1) * (bw * 1e9 / freq) * 2  # 50% probability of turn; once per cycle
            num_ubump += 2

    for _, l in set(xy_resurface_reg):
        power_ubump += 0.64 * 1e-6 * (l * bw) * (bw * 1e9 / freq) * 2
        power_esd += 200 * 1e-15 * (1 * 1) * freq * (1 / 4) * l * (bw * 1e9 / freq) * 2
        num_ubump += 2

    for _, l in set(xy_resurface_turn):
        power_ubump += 0.64 * 1e-6 * (l * bw) * (bw * 1e9 / freq)
        power_esd += 200 * 1e-15 * (1 * 1) * freq * (1 / 4) * l * (bw * 1e9 / freq)
        num_ubump += 1

    # get bridge chiplet number
    for pr, xy in xy_pr.items():
        if pr >= pnum and not is_covered_cpl(csys=csys, placement=placement, x=xy[0], y=xy[1]):
            xy_bgcpl.append(xy_pr[pr])

    for xy, _ in (xy_resurface_reg + xy_resurface_turn):
        if xy not in xy_bgcpl and not is_covered_cpl(csys=csys, placement=placement, x=xy[0], y=xy[1]):
            xy_bgcpl.append(xy)

    return {
        "power": power_router + power_wire,
        "power_eu": power_router + power_wire + power_esd + power_ubump,
        "pcost": num_ubump,
        "pcost_detail": {
            "num_ubump": num_ubump,
            "num_bgcpl": len(xy_bgcpl)
        },
        "perf": avg_lat_packet + lat_penal,
        "mlayout": (xy_pr, paths)
    }