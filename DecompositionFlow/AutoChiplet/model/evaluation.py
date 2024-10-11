from functools import reduce
from itertools import accumulate
import math
from typing import List, Dict, Set, Tuple
from .model import Block, Chiplet, Package, OS, SI
import networkx as nx
import numpy as np
from . import spec
import pandas as pd
from .. import utils


def make_package(type_pkg: str, chiplets: Dict[Chiplet, int]) -> Package:
    assert sum(chiplets.values())  # should not be empty
    if type_pkg == "OS":
        return OS(chiplets=chiplets)
    elif type_pkg == "SI":
        if sum(chiplets.values()) == 1:  # One chiplet, no need for interposer
            return OS(chiplets=chiplets)
        else:
            return SI(chiplets=chiplets)
    else:
        raise ValueError("Error: Invalid package type")


def graph_partition(g: nx.DiGraph, n: int, w_name="comm"):
    """
        Min-cut partitioning a graph using kernighan Lin algorithm
        g: the graph to be partitioned
        n: the number of each partition
        
        ref: https://eecs.wsu.edu/~daehyun/teaching/2015_EE582/ppt/02-partitioning.pdf
             http://users.ece.northwestern.edu/~haizhou/357/lec2.pdf
    """
    g_ = g.to_undirected()  # deep copy
    for u, v, attr in g_.edges(data=True):
        attr["w"] = g[u][v][w_name] if g.has_edge(u, v) else 0
        attr["w"] += g[v][u][w_name] if g.has_edge(v, u) else 0

    bid: Dict[Block, List[int]] = {}  # bid: node id/name of each type of block
    bn: Dict[Block, int] = {}  # bn: number of each type of block in one chiplet
    for v, attr_v in g_.nodes(data=True):
        blk = attr_v["block"]
        if attr_v["block"] not in bid:
            bid[blk] = []
        bid[blk].append(v)
    for blk in bid:
        assert len(bid[blk]) % n == 0
        bn[blk] = len(bid[blk]) // n

    # generate initial partitioning
    ptt: List[List[int]] = []  # partition
    for i in range(n):
        p = []
        for blk, num_blk in bn.items():
            p.extend(bid[blk][i * num_blk:(i + 1) * num_blk])
        ptt.append(p)

    # if n >= 5:  # TODO: when n is large, the partition is almost useless
    # return ptt
    exit_cnt = 0
    while exit_cnt < 5:
        exit_cnt += 1
        flg_imprv = False
        for idx_i in range(len(ptt)):
            for idx_j in range(idx_i + 1, len(ptt)):
                set_i, set_j = set(ptt[idx_i]), set(ptt[idx_j])
                # compute D of ptt[idx_i]
                for v_i in ptt[idx_i]:
                    neigh_g = set(g_.neighbors(v_i))
                    E, I = 0, 0
                    for w_j in (neigh_g - set_i) & set_j:  # external cost
                        E += g_[v_i][w_j]["w"] if g_.has_edge(v_i, w_j) else 0
                    for w_i in neigh_g & set_i:  # internal cost
                        I += g_[v_i][w_i]["w"] if g_.has_edge(v_i, w_i) else 0
                    g_.nodes[v_i]["D"] = E - I
                # compute D of ptt[idx_j]
                for v_j in ptt[idx_j]:
                    neigh_g = set(g_.neighbors(v_j))
                    E, I = 0, 0
                    for w_i in (neigh_g - set_j) & set_i:  # external cost
                        E += g_[v_j][w_i]["w"] if g_.has_edge(v_j, w_i) else 0
                    for w_j in neigh_g & set_j:  # internal cost
                        I += g_[v_j][w_j]["w"] if g_.has_edge(v_j, w_j) else 0
                    g_.nodes[v_j]["D"] = E - I

                nx.set_node_attributes(g_, False, name="lock")
                queue, ga_hist = [], []
                while True:
                    gain = []
                    for u in ptt[idx_i]:
                        for v in ptt[idx_j]:
                            if g_.nodes[u]["block"] == g_.nodes[v][
                                    "block"] and not g_.nodes[u]["lock"] and not g_.nodes[v]["lock"]:
                                gain.append(
                                    ((u, v),
                                     g_.nodes[u]["D"] + g_.nodes[v]["D"] - 2 * (g_[u][v]["w"] if g_.has_edge(u, v) else 0)))
                    if len(gain):
                        (u_sel, v_sel), ga = max(gain, key=lambda e: e[1])
                        queue.append((u_sel, v_sel))
                        ga_hist.append(ga)
                        g_.nodes[u_sel]["lock"] = True
                        g_.nodes[v_sel]["lock"] = True
                        # update D
                        for x in ptt[idx_i]:
                            if x != u_sel and not g_.nodes[x]["lock"]:
                                g_.nodes[x]["D"] += 2 * (g_[x][u_sel]["w"] if g_.has_edge(x, u_sel) else
                                                         0) - 2 * (g_[x][v_sel]["w"] if g_.has_edge(x, v_sel) else 0)
                        for y in ptt[idx_j]:
                            if y != v_sel and not g_.nodes[y]["lock"]:
                                g_.nodes[y]["D"] += 2 * (g_[y][v_sel]["w"] if g_.has_edge(y, v_sel) else
                                                         0) - 2 * (g_[y][u_sel]["w"] if g_.has_edge(y, u_sel) else 0)
                    else:
                        if not len(ga_hist):  # available swapping done
                            break
                        ga_hist = list(accumulate(ga_hist))
                        k = np.argmax(ga_hist)
                        if ga_hist[k] <= 0:
                            break
                        flg_imprv = True
                        for u, v in queue[:k + 1]:  # perform swapping
                            ptt[idx_i].remove(u)
                            ptt[idx_i].append(v)
                            ptt[idx_j].remove(v)
                            ptt[idx_j].append(u)
                        queue, ga_hist = [], []
        if not flg_imprv:
            break
    return ptt


def get_cost(pkgs: List[Package], vols: List[int], nbd: Dict[Block, int] = None, ncd: Dict[Chiplet, int] = None):
    """
        The cost evaluation can be operated on multiple packages due to chiplet reuse
        nbd: total number of block dict
        ncd: total number of chiplet dict
    """
    assert len(pkgs) == len(vols)
    if ncd is None:  # use external ncd to take unseen chiplets into account
        ncd = {}
        for pkg, vol in zip(pkgs, vols):
            for cpl, num_cpl in pkg.chiplets.items():
                if cpl not in ncd:
                    ncd[cpl] = 0
                ncd[cpl] += vol * num_cpl

    if nbd is None:
        nbd = {}
        for pkg, vol in zip(pkgs, vols):
            for cpl, num_cpl in pkg.chiplets.items():
                for blk, num_blk in cpl.blocks.items():
                    if blk not in nbd:
                        nbd[blk] = 0
                    nbd[blk] += vol * num_cpl * num_blk

    cost: List[float] = []
    for pkg, vol in zip(pkgs, vols):
        cost_sys = sum(pkg.RE()) + pkg.NRE() / vol
        for cpl, num_cpl in pkg.chiplets.items():
            cost_sys += cpl.NRE() / ncd[cpl] * num_cpl
            for blk, num_blk in cpl.blocks.items():
                cost_sys += blk.NRE() / nbd[blk] * num_blk * num_cpl
        cost.append(cost_sys)
    return cost


def get_pp(bdg: nx.DiGraph, pm: np.ndarray):
    """
        Return (power, performance)
        pm: partition matrix
    """
    assert len(bdg) == pm.shape[0] == pm.shape[1]
    power = 0
    perf = 0
    for u, v, e_attr in bdg.edges(data=True):
        assert u != v  # should not have self-loop
        if not pm[u][v]:  # 0 represents block u, v are in different chiplets
            assert not pm[v][u]
            power += e_attr["comm"] * e_attr["ener_eff"] * 1e-3 * 8
            perf += e_attr["perf_penal"]  # a rough linear model
    return power, perf


def get_pp_final(bdg: nx.DiGraph, ptt: List[List[int]]):
    """
        Return (power, performance)
        pm: partition matrix
    """
    assert sum(map(len, ptt)) == len(bdg)

    num_cpl = len(ptt)
    traffic_total = sum([attr["comm"] for _, __, attr in bdg.edges(data=True)]) * 8
    areas = [0] * num_cpl
    connection_matrix = [[0] * num_cpl for _ in range(num_cpl)]  # number of wire from chiplet i to chiplet j
    traffic_matrix = np.zeros(shape=(num_cpl, num_cpl))  # traffic (Gbit/s) from chiplet i to chiplet j

    for i in range(len(ptt)):
        for j in range(len(ptt[i])):
            areas[i] += bdg.nodes[ptt[i][j]]["block"].area

    for i_a in range(len(ptt)):
        for j_a in range(len(ptt[i_a])):
            for i_b in range(len(ptt)):
                for j_b in range(len(ptt[i_b])):
                    if i_a != i_b and bdg.has_edge(ptt[i_a][j_a], ptt[i_b][j_b]):
                        connection_matrix[i_a][i_b] += math.ceil(bdg[ptt[i_a][j_a]][ptt[i_b][j_b]]["comm"] * 8)
                        traffic_matrix[i_a][i_b] += bdg[ptt[i_a][j_a]][ptt[i_b][j_b]]["comm"] * 8

    wl_matrix = utils.place(areas=areas, connection_matrix=connection_matrix)

    power = 0
    perf = 0

    lmax_cycle = 10  # maximum distance per cycle
    ener_eff_io = 0.59
    ener_eff_wl = lambda wirelength: 2.583171 / 64 * wirelength
    ener_eff_reg = 2.150397 / 128
    for i in range(num_cpl):
        for j in range(num_cpl):
            power += traffic_matrix[i][j] * (ener_eff_io + ener_eff_wl(wl_matrix[i][j]) +
                                             math.floor(wl_matrix[i][j] / lmax_cycle) * ener_eff_reg) / 1000
            perf += (2 + math.floor(wl_matrix[i][j] / lmax_cycle)) * traffic_matrix[i][j] / traffic_total

    return power, perf


def eval_ev_ppc(bdg_all: List[nx.DiGraph],
                vol_all: List[int],
                w_power: float,
                w_perf: float,
                w_cost: float,
                type_pkg: str,
                bc_all: List[List[Set[Block]]] = None,
                cn_all: List[List[int]] = None,
                bid_all: List[Dict[Block, List[int]]] = None,
                ptt_all: List[List[List[int]]] = None):
    """
        Return (energy value, ppc of each system) with D2D overhead consideration.
        ptt_all: None when bc_all/cn_all/bid_all is not None
    """
    def make_chiplet(bdg: nx.DiGraph, p: Set[int]):
        """
            p: set of node id in bdg
        """
        blocks: Dict[Block] = {}
        for n in p:
            blk = bdg.nodes[n]["block"]
            if blk not in blocks:
                blocks[blk] = 0
            blocks[blk] += 1
            cpl = Chiplet(blocks=blocks)
        return cpl

    power, perf, cost = [], [], []

    pm_all = []
    if ptt_all is None:
        assert len(bdg_all) == len(bc_all) == len(cn_all)
        for idx_sys, (bc_sys, cn_sys) in enumerate(zip(bc_all, cn_all)):  # evaluate all systems
            pm = np.zeros(shape=(len(bdg_all[idx_sys]), len(bdg_all[idx_sys])), dtype=int)
            ptt_ = []
            assert len(bc_sys) == len(cn_sys)
            for bc, cn in zip(bc_sys, cn_sys):
                nid_bc = reduce(list.__add__, [bid_all[idx_sys][blk] for blk in bc])
                g_bc = nx.subgraph(bdg_all[idx_sys], nid_bc)
                ptt = graph_partition(g=g_bc, n=cn)
                for p in ptt:
                    for i in p:
                        for j in p:
                            pm[i][j] = 1  # same partition, same chiplet
                ptt_.extend(ptt)
            pm_all.append(pm)
            po, pe = get_pp_final(bdg_all[idx_sys], ptt_)
            power.append(po)
            perf.append(pe)
    else:
        assert bc_all is None and cn_all is None and bid_all is None
        for idx_sys, (bdg, ptt) in enumerate(zip(bdg_all, ptt_all)):
            pm = np.zeros(shape=(len(bdg), len(bdg)), dtype=int)
            for p in ptt:
                for i in p:
                    for j in p:
                        pm[i][j] = 1  # same partition, same chiplet
            pm_all.append(pm)
            po, pe = get_pp_final(bdg_all[idx_sys], ptt)
            power.append(po)
            perf.append(pe)

    cpl_all = []
    cpl_comm: Dict[Chiplet, Tuple[float, float]] = {}  # (input, output) traffic
    for bdg, pm in zip(bdg_all, pm_all):
        g = nx.from_numpy_array(pm)
        ptt_sys: List[List[int]] = []
        cpl_sys: List[Chiplet] = []
        for comp in nx.connected_components(g):
            ptt_sys.append(list(comp))
            cpl_sys.append(make_chiplet(bdg=bdg, p=comp))
        cpl_all.append(cpl_sys)

        for i in range(len(ptt_sys)):
            comm_in, comm_out = 0, 0  # comm for ptt_sys[i]
            for j in range(len(ptt_sys)):
                if i == j:
                    continue
                for n_i in ptt_sys[i]:
                    for n_j in ptt_sys[j]:
                        if bdg.has_edge(n_i, n_j):
                            comm_out += bdg[n_i][n_j]["comm"]
                        if bdg.has_edge(n_j, n_i):
                            comm_in += bdg[n_j][n_i]["comm"]
            cpl_i = cpl_sys[i]
            if cpl_i not in cpl_comm:
                cpl_comm[cpl_i] = (comm_in, comm_out)
            else:
                if (spec.D2D_symmetric and max((comm_in, comm_out)) > max(cpl_comm[cpl_i])) or (not spec.D2D_symmetric and sum(
                    (comm_in, comm_out)) > sum(cpl_comm[cpl_i])):
                    cpl_comm[cpl_i] = (comm_in, comm_out)

    # update the comm and generate pacakages
    pkg_all = []
    for cpl_sys in cpl_all:
        cpls: Dict[Chiplet, int] = {}
        for cpl in cpl_sys:
            if cpl not in cpls:
                cpl.set_comm(comm=cpl_comm[cpl])
                cpls[cpl] = 0
            cpls[cpl] += 1
        pkg_all.append(make_package(type_pkg=type_pkg, chiplets=cpls))

    ncd = {}
    for pkg, vol in zip(pkg_all, vol_all):
        for cpl, num_cpl in pkg.chiplets.items():
            if cpl not in ncd:
                ncd[cpl] = 0
            ncd[cpl] += vol * num_cpl
    nbd = {}
    for pkg, vol in zip(pkg_all, vol_all):
        for cpl, num_cpl in pkg.chiplets.items():
            for blk, num_blk in cpl.blocks.items():
                if blk not in nbd:
                    nbd[blk] = 0
                nbd[blk] += vol * num_cpl * num_blk

    cost: List[float] = []
    for pkg, vol in zip(pkg_all, vol_all):
        cost_sys = sum(pkg.RE()) + pkg.NRE() / vol
        for cpl, num_cpl in pkg.chiplets.items():
            cost_sys += cpl.NRE() / ncd[cpl] * num_cpl
            for blk, num_blk in cpl.blocks.items():
                cost_sys += blk.NRE() / nbd[blk] * num_blk * num_cpl
        cost.append(cost_sys)

    ev = w_power * np.average(power) + w_perf * np.average(perf) + w_cost * np.average(cost)
    ppc = list(zip(power, perf, cost))
    return ev, ppc


def get_cost_detail(bdg_all: List[nx.DiGraph],
                    vol_all: List[int],
                    type_pkg: str,
                    bc_all: List[List[Set[Block]]] = None,
                    cn_all: List[List[int]] = None,
                    bid_all: List[Dict[Block, List[int]]] = None,
                    ptt_all: List[List[List[int]]] = None):
    """
        Return detailed composition of cost.
    """
    def make_chiplet(bdg: nx.DiGraph, p: Set[int]):
        """
            p: set of node id in bdg
        """
        blocks: Dict[Block] = {}
        for n in p:
            blk = bdg.nodes[n]["block"]
            if blk not in blocks:
                blocks[blk] = 0
            blocks[blk] += 1
            cpl = Chiplet(blocks=blocks)
        return cpl

    cost = [], []

    pm_all = []
    if ptt_all is None:
        assert len(bdg_all) == len(bc_all) == len(cn_all)
        for idx_sys, (bc_sys, cn_sys) in enumerate(zip(bc_all, cn_all)):  # evaluate all systems
            pm = np.zeros(shape=(len(bdg_all[idx_sys]), len(bdg_all[idx_sys])), dtype=int)
            assert len(bc_sys) == len(cn_sys)
            for bc, cn in zip(bc_sys, cn_sys):
                nid_bc = reduce(list.__add__, [bid_all[idx_sys][blk] for blk in bc])
                g_bc = nx.subgraph(bdg_all[idx_sys], nid_bc)
                ptt = graph_partition(g=g_bc, n=cn)
                for p in ptt:
                    for i in p:
                        for j in p:
                            pm[i][j] = 1  # same partition, same chiplet
            pm_all.append(pm)
    else:
        assert bc_all is None and cn_all is None and bid_all is None
        for bdg, ptt in zip(bdg_all, ptt_all):
            pm = np.zeros(shape=(len(bdg), len(bdg)), dtype=int)
            for p in ptt:
                for i in p:
                    for j in p:
                        pm[i][j] = 1  # same partition, same chiplet
            pm_all.append(pm)

    cpl_all = []
    cpl_comm: Dict[Chiplet, Tuple[float, float]] = {}  # (input, output) traffic
    for bdg, pm in zip(bdg_all, pm_all):
        g = nx.from_numpy_array(pm)
        ptt_sys: List[List[int]] = []
        cpl_sys: List[Chiplet] = []
        for comp in nx.connected_components(g):
            ptt_sys.append(list(comp))
            cpl_sys.append(make_chiplet(bdg=bdg, p=comp))
        cpl_all.append(cpl_sys)

        for i in range(len(ptt_sys)):
            comm_in, comm_out = 0, 0  # comm for ptt_sys[i]
            for j in range(len(ptt_sys)):
                if i == j:
                    continue
                for n_i in ptt_sys[i]:
                    for n_j in ptt_sys[j]:
                        if bdg.has_edge(n_i, n_j):
                            comm_out += bdg[n_i][n_j]["comm"]
                        if bdg.has_edge(n_j, n_i):
                            comm_in += bdg[n_j][n_i]["comm"]
            cpl_i = cpl_sys[i]
            if cpl_i not in cpl_comm:
                cpl_comm[cpl_i] = (comm_in, comm_out)
            else:
                if (spec.D2D_symmetric and max((comm_in, comm_out)) > max(cpl_comm[cpl_i])) or (not spec.D2D_symmetric and sum(
                    (comm_in, comm_out)) > sum(cpl_comm[cpl_i])):
                    cpl_comm[cpl_i] = (comm_in, comm_out)

    # update the comm and generate pacakages
    pkg_all = []
    for cpl_sys in cpl_all:
        cpls: Dict[Chiplet, int] = {}
        for cpl in cpl_sys:
            if cpl not in cpls:
                cpl.set_comm(comm=cpl_comm[cpl])
                cpls[cpl] = 0
            cpls[cpl] += 1
        pkg_all.append(make_package(type_pkg=type_pkg, chiplets=cpls))

    ncd = {}
    for pkg, vol in zip(pkg_all, vol_all):
        for cpl, num_cpl in pkg.chiplets.items():
            if cpl not in ncd:
                ncd[cpl] = 0
            ncd[cpl] += vol * num_cpl
    if True:  # get data
        print("****", ncd, "****")

    nbd = {}
    for pkg, vol in zip(pkg_all, vol_all):
        for cpl, num_cpl in pkg.chiplets.items():
            for blk, num_blk in cpl.blocks.items():
                if blk not in nbd:
                    nbd[blk] = 0
                nbd[blk] += vol * num_cpl * num_blk
    """
        NRE Cost of Packages
        NRE cost of SoC Chips/Chiplets
        
        RE Cost of Raw Packages
        RE Cost of Package Defects
        RE Cost of Raw Dies
        RE Cost of Die Defects
        RE Cost of Wasted KGDs
    """

    cost: List[float] = []
    cost_detail = []
    for pkg, vol in zip(pkg_all, vol_all):
        cd_sys = {}
        cost_sys = sum(pkg.RE()) + pkg.NRE() / vol
        cd_sys["NRE Cost of Packages"] = pkg.NRE() / vol
        cd_sys["NRE cost of SoC Chips/Chiplets"] = 0
        (cost_raw_chiplets, cost_defect_chiplets, cost_raw_package, cost_defect_package, cost_wasted_chiplets) = pkg.RE()
        cd_sys["RE Cost of Raw Packages"] = cost_raw_package
        cd_sys["RE Cost of Package Defects"] = cost_defect_package
        cd_sys["RE Cost of Raw Dies"] = cost_raw_chiplets
        cd_sys["RE Cost of Die Defects"] = cost_defect_chiplets
        cd_sys["RE Cost of Wasted KGDs"] = cost_wasted_chiplets

        for cpl, num_cpl in pkg.chiplets.items():
            cost_sys += cpl.NRE() / ncd[cpl] * num_cpl
            cd_sys["NRE cost of SoC Chips/Chiplets"] += cpl.NRE() / ncd[cpl] * num_cpl
            for blk, num_blk in cpl.blocks.items():
                cost_sys += blk.NRE() / nbd[blk] * num_blk * num_cpl
                cd_sys["NRE cost of SoC Chips/Chiplets"] += blk.NRE() / nbd[blk] * num_blk * num_cpl
        cost.append(cost_sys)
        cost_detail.append(cd_sys)
    return pd.DataFrame(cost_detail)


def get_bd_bid(bdg_all: List[nx.DiGraph]):
    bd_all: List[Dict[Block, int]] = []  # each item is block description of the SoC: {"cpu":2, "gpu":4}
    bid_all: List[Dict[Block, List[int]]] = []
    for bdg in bdg_all:
        bd = {}
        bid = {}
        for nid, attr in bdg.nodes(data=True):
            blk = attr["block"]
            if blk not in bd:
                bd[blk] = 0
                bid[blk] = []
            bd[blk] += 1
            bid[blk].append(nid)
        bd_all.append(bd)
        bid_all.append(bid)
    return bd_all, bid_all