import copy
from enum import Enum
import time

from bidict import bidict
from ..GIA import *
from ..ChipletSys import gen_csys
from rectpack import newPacker
import argparse
from tqdm import tqdm, trange
from multiprocessing import Pool
from ..Sunfloor import sunfloor
import pickle
from ..NoC import show_graph
import os
from .. import NoC

dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
dir_dsent = os.path.join(dir_root, "tool/dsent")
dir_tgff = os.path.join(dir_root, "tool/tgff")
dir_chaco = os.path.join(dir_root, "tool/chaco")
dir_hotspot = os.path.join(dir_root, "tool/hotspot")
dir_booksim = os.path.join(dir_root, "tool/booksim")


def test_find_path1():
    intp_graph = nx.DiGraph()
    size_map = 50
    for (x_u, x_v) in zip(range(0, size_map), range(1, size_map)):
        for y in range(size_map):
            intp_graph.add_edge((x_u, y), (x_v, y))
            intp_graph.add_edge((x_v, y), (x_u, y))

    for (y_u, y_v) in zip(range(0, size_map), range(1, size_map)):
        for x in range(size_map):
            intp_graph.add_edge((x, y_u), (x, y_v))
            intp_graph.add_edge((x, y_v), (x, y_u))

    srcs = []
    dsts = []

    for _ in range(80):
        while True:
            src = tuple(np.random.randint(low=0, high=size_map, size=2))
            dst = tuple(np.random.randint(low=0, high=size_map, size=2))
            if src != dst:
                srcs.append(src)
                dsts.append(dst)
                break

    paths = find_path(intp_graph=intp_graph, sd=list(zip(srcs, dsts)), bend_cost=8)
    show_intp(size_map, size_map, paths)


def test_find_path2():
    intp_graph = nx.DiGraph()
    size_map = 50
    for (x_u, x_v) in zip(range(0, size_map), range(1, size_map)):
        for y in range(size_map):
            intp_graph.add_edge((x_u, y), (x_v, y))
            intp_graph.add_edge((x_v, y), (x_u, y))

    for (y_u, y_v) in zip(range(0, size_map), range(1, size_map)):
        for x in range(size_map):
            intp_graph.add_edge((x, y_u), (x, y_v))
            intp_graph.add_edge((x, y_v), (x, y_u))

    srcs = [(0, 0)]
    dsts = [(0, 0)]
    paths = find_path(intp_graph=intp_graph, sd=list(zip(srcs, dsts)), bend_cost=8)
    show_intp(size_map, size_map, paths)


def test_find_path3():
    intp_graph = nx.DiGraph()

    intp_graph.add_edge((0, 0), (0, 1))
    intp_graph.add_edge((0, 1), (0, 2))
    intp_graph.add_edge((0, 2), (0, 3))
    intp_graph.add_edge((0, 3), (1, 3))
    intp_graph.add_edge((1, 3), (2, 3))
    intp_graph.add_edge((2, 3), (3, 3))
    intp_graph.add_edge((3, 3), (3, 2))

    intp_graph.add_edge((0, 0), (1, 0))
    intp_graph.add_edge((1, 0), (1, 1))
    intp_graph.add_edge((1, 1), (2, 1))
    intp_graph.add_edge((2, 1), (2, 2))
    intp_graph.add_edge((2, 2), (3, 2))
    intp_graph.add_edge((3, 2), (3, 3))

    size_map = 4
    paths = find_path(intp_graph=intp_graph, sd=[((0, 0), (3, 2))], bend_cost=1.001)
    print(paths)
    paths = find_path(intp_graph=intp_graph, sd=[((0, 0), (3, 2))], bend_cost=0.999)
    print(paths)


def test_search_aster4():
    """
        test y_penal
        o o o s—o o
              | |  
        o—o—o—o o—o
        |         |
        o—o—o—o—d o
                | |
        o o o o o—o
    """
    intp_graph = nx.DiGraph()

    intp_graph.add_edge((3, 3), (3, 2))
    intp_graph.add_edge((3, 2), (2, 2))
    intp_graph.add_edge((2, 2), (1, 2))
    intp_graph.add_edge((1, 2), (0, 2))
    intp_graph.add_edge((0, 2), (0, 1))
    intp_graph.add_edge((0, 1), (1, 1))
    intp_graph.add_edge((1, 1), (2, 1))
    intp_graph.add_edge((2, 1), (3, 1))
    intp_graph.add_edge((3, 1), (4, 1))

    intp_graph.add_edge((3, 3), (4, 3))
    intp_graph.add_edge((4, 3), (4, 2))
    intp_graph.add_edge((4, 2), (5, 2))
    intp_graph.add_edge((5, 2), (5, 1))
    intp_graph.add_edge((5, 1), (5, 0))
    intp_graph.add_edge((5, 0), (4, 0))
    intp_graph.add_edge((4, 0), (4, 1))

    size_map = 6
    paths = find_path(intp_graph=intp_graph, sd=[((3, 3), (4, 1)), ((3, 3), (4, 1))], bend_cost=0.5, y_penal=0.50001)
    print(paths)
    paths = find_path(intp_graph=intp_graph, sd=[((3, 3), (4, 1)), ((3, 3), (4, 1))], bend_cost=0.5, y_penal=0.49999)
    print(paths)


def test_rectpack():
    W_intp = 50
    H_intp = 50
    max_area_sys = 0.5 * W_intp * H_intp
    csystems = gen_csys(dir_tgff=dir_tgff,
                        bw=128,
                        cfg_tgff={
                            "task_cnt": (50, 5),
                            "seed": 0,
                            "tg_cnt": 100
                        },
                        filter=None,
                        clean=False,
                        cfg_csys={
                            "W_intp": W_intp,
                            "H_intp": H_intp,
                            "max_size_cpl": 20,
                            "max_wh_ratio_cpl": 2,
                            "max_area_sys": max_area_sys,
                            "max_num_cpl": 20,
                            "min_num_cpl": 10,
                            "max_pd_cpl": 0.8,
                            "min_pd_cpl": 0.2,
                            "max_pd_sys": 0.5,
                            "min_pd_sys": 0.38,
                            "max_incr_size": 50
                        })

    for csys in csystems:
        packer = newPacker()
        for idx_cpl, cpl in enumerate(csys.chiplets):
            packer.add_rect(width=cpl.w_orig, height=cpl.h_orig, rid=idx_cpl)
        packer.add_bin(width=W_intp, height=H_intp, count=1)
        packer.pack()
        all_rects = packer.rect_list()

        placement = [0] * len(csys.chiplets)
        for rect in all_rects:
            _, x, y, w, h, rid = rect
            assert (w == csys.chiplets[rid].w_orig and h == csys.chiplets[rid].h_orig) or (h == csys.chiplets[rid].w_orig
                                                                                           and w == csys.chiplets[rid].h_orig)
            angle = 0 if (w == csys.chiplets[rid].w_orig and h == csys.chiplets[rid].h_orig) else 1
            placement[rid] = (x, y, angle)

        csys.show_placement(placement=placement)
        print("max temperature: ",
              csys.eval_thermal(dir_hotspot=dir_hotspot, tile_size=0.001, placement=placement, visualize=True).max())


class Direction(Enum):
    N = 0
    S = 1
    W = 2
    E = 3


def get_DIR(u, v):
    """
        return the direction pointing from u to v
    """
    x_u, y_u = u
    x_v, y_v = v
    assert abs(x_v - x_u) + abs(y_v - y_u) == 1
    if x_u == x_v:
        if y_u > y_v:
            return Direction.S
        else:
            return Direction.N
    else:
        assert y_u == y_v
        if x_u > x_v:
            return Direction.W
        else:
            return Direction.E


def test_gen_layout_active():
    dir_log_root = os.path.join(dir_root, "data/log_11_15")
    path_dt = os.path.join(dir_root, "data/dataset/sys_11_15_topt.pkl")
    with open(path_dt, "rb") as f:
        dt = pickle.load(f)  # data point
        for d in dt:
            csys = ChipletSys(W=d["W_intp"],
                              H=d["H_intp"],
                              chiplets=d["chiplets"],
                              task_graph=d["task_graph"],
                              pin_map=d["pin_map"])

            placement = d["placement"][0]
            topo_graph, rtp = d["topo_graph"], d["rtp"]
            time_beg = time.time()
            flg_suc, xy_pr, paths = gen_layout_active(csys=csys,
                                                    topo_graph=topo_graph,
                                                    rtp=rtp,
                                                    placement=placement,
                                                    faulted_links=None,
                                                    bend_cost=1,
                                                    max_retry=90)
            if flg_suc:
                print("success")
            else:
                print("fail")
            time_end = time.time()
            print(time_end - time_beg)

            cnum = len(csys.task_graph)
            cha_list = []
            turn = []
            for src, dst in topo_graph.edges():
                tile_beg = xy_pr[src]
                tile_end = xy_pr[dst]
                path_this = paths[(src, dst)]
                for idx_p, p in enumerate(path_this):
                    u, v, t = p
                    assert t == 0 or t == 1
                    if idx_p == 0:
                        assert u == tile_beg
                        if src >= cnum:
                            assert t == 0
                    elif idx_p == len(path_this) - 1:
                        assert v == tile_end
                        if dst >= cnum:
                            assert t == 0
                    cha_list.append(p)
                assert len(cha_list) == len(set(cha_list))

                for p_m, p_n in zip(path_this, path_this[1:]):
                    u_m, v_m, t_m = p_m
                    u_n, v_n, t_n = p_n
                    assert v_m == u_n
                    assert abs(u_m[0] - v_m[0]) + abs(u_m[1] - v_m[1]) == 1
                    assert abs(u_n[0] - v_n[0]) + abs(u_n[1] - v_n[1]) == 1

                    covered_by_cpl = False
                    x_v_m, y_v_m = v_m
                    for idx_c, (cx, cy, angle) in enumerate(placement):
                        if cx <= x_v_m <= cx + csys.chiplets[idx_c].w(angle) and cy <= y_v_m <= cy + csys.chiplets[idx_c].h(angle):
                            covered_by_cpl = True
                            break

                    if (get_DIR(u_m, v_m) != get_DIR(u_n, v_n) or t_m != t_n) and not covered_by_cpl:
                        turn.append(v_m)
            print(len(turn), len(set(turn)))


def test_gen_layout_passive():
    dir_log_root = os.path.join(dir_root, "data/log_11_15")
    path_dt = os.path.join(dir_root, "data/dataset/sys_11_15_topt.pkl")
    with open(path_dt, "rb") as f:
        dt = pickle.load(f)  # data point
        for idx_d, d in enumerate(dt):
            csys = ChipletSys(W=d["W_intp"],
                              H=d["H_intp"],
                              chiplets=d["chiplets"],
                              task_graph=d["task_graph"],
                              pin_map=d["pin_map"])
            placement = d["placement"][0]
            # csys.show_placement(placement)
            topo_graph = d["topo_graph"]
            rtp = d["rtp"]

            time_beg = time.time()
            flg_suc, xy_pr, paths = gen_layout_passive(csys=csys,
                                                       topo_graph=topo_graph,
                                                       rtp=rtp,
                                                       placement=placement,
                                                       faulted_links=None,
                                                       bend_cost=1,
                                                       y_cost=1,
                                                       max_retry=90)
            if flg_suc:
                print("success")
            else:
                print("fail")
                continue
            time_end = time.time()
            print(time_end - time_beg)

            cnum = len(csys.task_graph)
            cha_list = []
            turn = []
            for src, dst in topo_graph.edges():
                tile_beg = xy_pr[src]
                tile_end = xy_pr[dst]
                path_this = paths[(src, dst)]
                for idx_p, p in enumerate(path_this):
                    u, v, t = p
                    assert t == 0 or t == 1
                    if idx_p == 0:
                        assert u == tile_beg
                        if src >= cnum:
                            assert t == 0
                    elif idx_p == len(path_this) - 1:
                        assert v == tile_end
                        if dst >= cnum:
                            assert t == 0
                    cha_list.append(p)
                assert len(cha_list) == len(set(cha_list))

                for p_m, p_n in zip(path_this, path_this[1:]):
                    u_m, v_m, t_m = p_m
                    u_n, v_n, t_n = p_n
                    assert v_m == u_n
                    assert abs(u_m[0] - v_m[0]) + abs(u_m[1] - v_m[1]) == 1
                    assert abs(u_n[0] - v_n[0]) + abs(u_n[1] - v_n[1]) == 1

                    covered_by_cpl = False
                    x_v_m, y_v_m = v_m
                    for idx_c, (cx, cy, angle) in enumerate(placement):
                        if cx <= x_v_m <= cx + csys.chiplets[idx_c].w(angle) and cy <= y_v_m <= cy + csys.chiplets[idx_c].h(
                                angle):
                            covered_by_cpl = True

                    if (get_DIR(u_m, v_m) != get_DIR(u_n, v_n) or t_m != t_n) and not covered_by_cpl:
                        turn.append(v_m)
            print(len(turn), len(set(turn)))



"""
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
        rtp_new[(s, d)] = path_new[1:-1]

    return rtp_new


def check_topo(topo_graph, core_num):
    for n in topo_graph:
        if n < core_num:  # is a core
            neighbor = set.union(set(topo_graph.predecessors(n)), set(topo_graph.successors(n)))
            if len(neighbor) > 1:
                raise ValueError("ERROR: core {} connects to more than 1 core/router:{}".format(n, topo_graph.edges()))


def test_gen_layout_sip():
    dir_log_root = os.path.join(dir_root, "data/log_11_15")
    path_dt = os.path.join(dir_root, "data/dataset/sys_11_15_topt.pkl")
    csystems = []
    with open(path_dt, "rb") as f:
        dt = pickle.load(f)  # data point
        lat_S_P = []  # sip to passive GIA
        for idx_d, d in enumerate(dt):
            csys = ChipletSys(W=d["W_intp"],
                              H=d["H_intp"],
                              chiplets=d["chiplets"],
                              task_graph=d["task_graph"],
                              pin_map=d["pin_map"])
            placement = d["placement"][0]
            # csys.show_placement(placement)
            topo_graph, rtp = d["topo_graph"], d["rtp"]

            PPA = NoC.eval_PPA_booksim(dir_booksim=dir_booksim,
                                       task_graph=csys.task_graph,
                                       topo_graph=topo_graph,
                                       rtp=rtp,
                                       cfg={
                                           "sim_cycle": 10000,
                                           "num_vcs": 4,
                                           "vc_buf_size": 4
                                       },
                                       clean=True)
            avg_lat_packet_old = PPA["avg_lat_packet"]

            flg_suc, xy_pr, paths = gen_layout_sip(csys=csys,
                                                   topo_graph=topo_graph,
                                                   rtp=rtp,
                                                   placement=placement,
                                                   faulted_links=None,
                                                   bend_cost=2,
                                                   y_cost=2,
                                                   max_retry=45)
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

                                paths_new = dict([(k, v) for (k, v) in paths.items() if k != (s1, d1) and k != (s2, d2)])
                                topo_graph.remove_edge(s1, d1)
                                topo_graph.remove_edge(s2, d2)
                                # print("remove ({}, {}) and ({}, {})".format(s1, d1, s2, d2))

                                if rtr_in != s1:
                                    topo_graph.add_edge(s1, rtr_in)
                                    paths_new[(s1, rtr_in)] = paths[(s1, d1)][:idx_p1]

                                if rtr_in != s2:
                                    topo_graph.add_edge(s2, rtr_in)
                                    paths_new[(s2, rtr_in)] = paths[(s2, d2)][:idx_p2]

                                # if not (rtr_in == s1 and rtr_out == d1) or not (rtr_in == s2 and rtr_out == d2):
                                topo_graph.add_edge(rtr_in, rtr_out)
                                paths_new[(rtr_in, rtr_out)] = paths[(s1, d1)][idx_p1:idx_p1 + length]
                                if not paths[(s1, d1)][idx_p1:idx_p1 + length]:
                                    import pdb
                                    pdb.set_trace()

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
                        elif length:
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
                                check_topo(topo_graph, pnum)
                                topo_graph.remove_edge(s2, d2)
                                check_topo(topo_graph, pnum)
                                rev1 = s1 < pnum or d1 < pnum and topo_graph.has_edge(d1, s1)  # need to reverse
                                rev2 = s2 < pnum or d2 < pnum and topo_graph.has_edge(d2, s2)
                                if rev1:
                                    if topo_graph.has_edge(d1, s1):
                                        topo_graph.remove_edge(d1, s1)
                                if rev2:
                                    if topo_graph.has_edge(d2, s2):
                                        topo_graph.remove_edge(d2, s2)

                                if rtr_in != s1:
                                    topo_graph.add_edge(s1, rtr_in)
                                    check_topo(topo_graph, pnum)
                                    if rev1:
                                        topo_graph.add_edge(rtr_in, s1)
                                        check_topo(topo_graph, pnum)
                                    paths_new[(s1, rtr_in)] = paths[(s1, d1)][:idx_p1]

                                if rtr_in != s2:
                                    if (s1, d1) != (d2, s2):
                                        topo_graph.add_edge(s2, rtr_in)
                                        check_topo(topo_graph, pnum)
                                        if rev2:
                                            topo_graph.add_edge(rtr_in, s2)
                                            check_topo(topo_graph, pnum)
                                        paths_new[(s2, rtr_in)] = paths[(s2, d2)][:idx_p2]

                                # if not (rtr_in == s1 and rtr_out == d1) or not (rtr_in == s2 and rtr_out == d2):
                                topo_graph.add_edge(rtr_in, rtr_out)
                                paths_new[(rtr_in, rtr_out)] = paths[(s1, d1)][idx_p1:idx_p1 + length]
                                if rev1 or rev2:
                                    topo_graph.add_edge(rtr_out, rtr_in)
                                    check_topo(topo_graph, pnum)

                                if rtr_out != d1:
                                    topo_graph.add_edge(rtr_out, d1)
                                    check_topo(topo_graph, pnum)
                                    if rev1:
                                        topo_graph.add_edge(d1, rtr_out)
                                        check_topo(topo_graph, pnum)
                                    paths_new[(rtr_out, d1)] = paths[(s1, d1)][idx_p1 + length:]

                                if rtr_out != d2:
                                    if (s1, d1) != (d2, s2):
                                        topo_graph.add_edge(rtr_out, d2)
                                        check_topo(topo_graph, pnum)
                                        if rev2:
                                            topo_graph.add_edge(d2, rtr_out)
                                            check_topo(topo_graph, pnum)
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

            # print(xy_r)
            # print(paths)
            # NoC.show_graph(topo_graph)
            try:
                PPA = NoC.eval_PPA_booksim(dir_booksim=dir_booksim,
                                           task_graph=csys.task_graph,
                                           topo_graph=topo_graph,
                                           rtp=rtp,
                                           cfg={
                                               "sim_cycle": 10000,
                                               "num_vcs": 4,
                                               "vc_buf_size": 4
                                           },
                                           clean=True)
            except:
                print("time out")
            else:
                avg_lat_packet_new = PPA["avg_lat_packet"]
                print(avg_lat_packet_old, avg_lat_packet_new)
                lat_S_P.append(avg_lat_packet_new / avg_lat_packet_old)
                print("average ratio: ", sum(lat_S_P) / len(lat_S_P))
        print("average ratio: ", lat_S_P)
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tests for GIA package.")
    parser.add_argument("test_func")
    args = parser.parse_args()
    eval((args.test_func) + "()")