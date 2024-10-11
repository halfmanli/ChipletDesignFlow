import math
import os
import shutil
import subprocess
import tempfile
from itertools import chain, combinations, product
from multiprocessing import Pool
from typing import Any, Dict, List, Set, Tuple

import cplex
import networkx as nx

from .. import utils
from ..model import (Block, Chiplet, eval_ev_ppc, get_bd_bid, get_cost,
                     get_cost_detail, make_package)
from ..solver import IndiSolver


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
        graph_file += '\n{} '.format(int(G.nodes[i]["block"].area * 10))  # vertex weight
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


def mincut_metis(dir_metis: str, G: nx.DiGraph, k: int, clean: bool = True):
    '''
        get k-mincut by using metis
        Return: partition list: [[node a, node b, ...], ..., [node x, node y, ...]]
        
        G: networkx graph, the weight name of is "comm"
    '''
    if not nx.is_directed(G):
        raise TypeError("ERROR: unsupported undirected graph")
    if min(G.nodes()) != 0:
        raise ValueError("ERROR: the name of nodes in graph should start from zero")
    if k > len(G) or k < 1:
        raise ValueError("ERROR: invalid number of cut")

    # Note k = 1 must not use metis, error otherwise
    if k == 1:
        return [list(G.nodes())]
    assert G.number_of_edges()

    e_cnt = 0
    graph_file = ''
    for i in G:
        assert not G.has_edge(i, i)  # has not self cycle
        graph_file += '\n{} '.format(int(G.nodes[i]["block"].area * 10))  # vertex weight
        for j in G:
            wgt_sum = 0
            wgt_sum += G.edges[(i, j)]["comm"] if G.has_edge(i, j) else 0  # metis only supports undirected edges
            wgt_sum += G.edges[(j, i)]["comm"] if G.has_edge(j, i) else 0
            if wgt_sum:
                graph_file += '%d %d ' % (j + 1, math.ceil(wgt_sum * 100))  # neighbor edgeweight. node starts from 1
                e_cnt += 1
    graph_file = '%d %d 011' % (len(G), e_cnt / 2) + graph_file  # e_cnt/2: count twice
    dir_tmp = tempfile.mkdtemp(dir=os.path.join(dir_metis, "build"))
    graph_path = os.path.join(dir_tmp, "graph.graph")
    res_path = "{}.part.{}".format(graph_path, k)
    with open(graph_path, "w") as f:
        f.write(graph_file)

    cmd_list = ["gpmetis", graph_path, str(k)]
    run_res = subprocess.run(cmd_list, capture_output=True, text=True, cwd=dir_tmp)
    try:
        with open(res_path, 'r') as r_f:
            res_raw = r_f.readlines()
            res = [[] for _ in range(k)]
            for idx_l, l in enumerate(res_raw):
                res[int(l)].append(idx_l)
    except:  # something goes wrong
        raise RuntimeError("ERROR: Metis runs failed!!!", "|||STDOUT:", run_res.stdout, "|||STDERR:", run_res.stderr)
    if clean:
        shutil.rmtree(dir_tmp)
    return res


def monolithic(bdg_all: List[nx.DiGraph], vol_all: List[int], params: Dict[str, Any]):
    type_pkg = "OS"
    w_power = params["w_power"]
    w_perf = params["w_perf"]
    w_cost = params["w_cost"]

    assert len(bdg_all) == len(vol_all)

    bd_all, bid_all = get_bd_bid(bdg_all=bdg_all)
    bc_all, cn_all = [], []
    for idx_sys in range(len(bdg_all)):
        bc_sys = [set(bd_all[idx_sys])]
        cn_sys = [1]
        bc_all.append(bc_sys)
        cn_all.append(cn_sys)
    ev, ppc = eval_ev_ppc(bdg_all=bdg_all,
                          vol_all=vol_all,
                          w_power=w_power,
                          w_perf=w_perf,
                          w_cost=w_cost,
                          type_pkg=type_pkg,
                          bc_all=bc_all,
                          cn_all=cn_all,
                          bid_all=bid_all)
    cost_detail = get_cost_detail(bdg_all=bdg_all,
                                  vol_all=vol_all,
                                  type_pkg=type_pkg,
                                  bc_all=bc_all,
                                  cn_all=cn_all,
                                  bid_all=bid_all)
    return ev, ppc, cost_detail


def finest_granularity(bdg_all: List[nx.DiGraph], vol_all: List[int], params: Dict[str, Any]):
    type_pkg = params["type_pkg"]
    w_power = params["w_power"]
    w_perf = params["w_perf"]
    w_cost = params["w_cost"]

    assert len(bdg_all) == len(vol_all)

    bd_all, bid_all = get_bd_bid(bdg_all=bdg_all)
    bc_all, cn_all = [], []
    for idx_sys in range(len(bdg_all)):
        bc_sys, cn_sys = [], []
        for blk in bd_all[idx_sys]:
            bc_sys.append({blk})
            cn_sys.append(bd_all[idx_sys][blk])
        bc_all.append(bc_sys)
        cn_all.append(cn_sys)
    ev, ppc = eval_ev_ppc(bdg_all=bdg_all,
                          vol_all=vol_all,
                          w_power=w_power,
                          w_perf=w_perf,
                          w_cost=w_cost,
                          type_pkg=type_pkg,
                          bc_all=bc_all,
                          cn_all=cn_all,
                          bid_all=bid_all)
    cost_detail = get_cost_detail(bdg_all=bdg_all,
                                  vol_all=vol_all,
                                  type_pkg=type_pkg,
                                  bc_all=bc_all,
                                  cn_all=cn_all,
                                  bid_all=bid_all)
    print(cn_all)
    return ev, ppc, cost_detail


def chopin(bdg_all: nx.DiGraph, vol_all: int, x: float, params: Dict[str, Any]):
    type_pkg = params["type_pkg"]
    w_power = params["w_power"]
    w_perf = params["w_perf"]
    w_cost = params["w_cost"]

    bd_all, bid_all = get_bd_bid(bdg_all=bdg_all)

    nbd = {}
    for bd_sys, vol_sys in zip(bd_all, vol_all):
        for blk in bd_sys:
            if blk not in nbd:
                nbd[blk] = 0
            nbd[blk] += bd_sys[blk] * vol_sys
    bc_all, cn_all = [], []
    for vol_sys, bd_sys in zip(vol_all, bd_all):
        blk_sys = list(bd_sys)
        bc_sys = [set([b]) for b in blk_sys]
        cn_cand_sys = [list(utils.find_factors(bd_sys[b])) for b in blk_sys]
        cn_combs = list(product(*cn_cand_sys))
        # for each combination, calculate cost accroding to paper
        res_sys = []
        for cn_sys in cn_combs:
            cpls: Dict[Chiplet, int] = {}
            for blk, cn in zip(blk_sys, cn_sys):
                cpl = Chiplet(blocks={blk: bd_sys[blk] // cn})
                assert cpl not in cpls
                cpls[cpl] = cn
            pkg = make_package(type_pkg=type_pkg, chiplets=cpls)
            cost = 0
            cost = sum(pkg.RE()) + pkg.NRE() / vol_sys
            for cpl, num_cpl in pkg.chiplets.items():
                cost += cpl.NRE() / (vol_sys * num_cpl)**x * num_cpl
                for blk, num_blk in cpl.blocks.items():
                    cost += blk.NRE() / nbd[blk] * num_blk * num_cpl
            res_sys.append((cost, bc_sys, cn_sys))
        _, bc_sys, cn_sys = min(res_sys, key=lambda e: e[0])
        bc_all.append(bc_sys)
        cn_all.append(cn_sys)
    print(cn_all)

    ev, ppc = eval_ev_ppc(bdg_all=bdg_all,
                          vol_all=vol_all,
                          w_power=w_power,
                          w_perf=w_perf,
                          w_cost=w_cost,
                          type_pkg=type_pkg,
                          bc_all=bc_all,
                          cn_all=cn_all,
                          bid_all=bid_all)
    cost_detail = get_cost_detail(bdg_all=bdg_all,
                                  vol_all=vol_all,
                                  type_pkg=type_pkg,
                                  bc_all=bc_all,
                                  cn_all=cn_all,
                                  bid_all=bid_all)
    return ev, ppc, cost_detail


def indp_opt(bdg: nx.DiGraph, vol: int, dir_log: str, params: Dict[str, Any]):
    isolver = IndiSolver(bdg=bdg, vol=vol, dir_log=dir_log, params=params)
    return isolver.opt()


def indp(bdg_all: List[nx.DiGraph], vol_all: List[int], params: Dict[str, Any]):
    """
        Opitmize system independently/individually without holistic optimization
    """
    dir_log_root = params["dir_log_root"]
    num_cpu = params["num_cpu"]  # number of process in pool
    pnum = params["pnum"]  # number of optimization process per bdg
    type_pkg = params["type_pkg"]
    w_power = params["w_power"]
    w_perf = params["w_perf"]
    w_cost = params["w_cost"]

    # individual optimization
    pool = Pool(processes=num_cpu)
    res_wait = []
    dir_log_indp = os.path.join(dir_log_root, "indp")
    os.mkdir(dir_log_indp)
    for idx_sys, (bdg, vol) in enumerate(zip(bdg_all, vol_all)):
        dir_log_indp_sys = os.path.join(dir_log_indp, str(idx_sys))
        os.mkdir(dir_log_indp_sys)
        res_wait_sys = []
        for pid in range(pnum):
            dir_log_indp_p = os.path.join(dir_log_indp_sys, str(pid))  # log of each optimization process
            os.mkdir(dir_log_indp_p)
            res_wait_sys.append(pool.apply_async(indp_opt, args=(bdg, vol, dir_log_indp_p, params)))
        res_wait.append(res_wait_sys)
    pool.close()
    pool.join()
    bc_all, cn_all = [], []
    for idx_sys in range(len(res_wait)):
        res_indi_sys = []
        for pid in range(pnum):
            res_indi_sys.append(res_wait[idx_sys][pid].get())
        _, (bc_sys, cn_sys) = min(res_indi_sys, key=lambda e: e[0])
        bc_all.append(bc_sys)
        cn_all.append(cn_sys)

    _, bid_all = get_bd_bid(bdg_all=bdg_all)
    ev, ppc = eval_ev_ppc(bdg_all=bdg_all,
                          vol_all=vol_all,
                          bc_all=bc_all,
                          cn_all=cn_all,
                          bid_all=bid_all,
                          w_power=w_power,
                          w_perf=w_perf,
                          w_cost=w_cost,
                          type_pkg=type_pkg)
    cost_detail = get_cost_detail(bdg_all=bdg_all,
                                  vol_all=vol_all,
                                  type_pkg=type_pkg,
                                  bc_all=bc_all,
                                  cn_all=cn_all,
                                  bid_all=bid_all)
    return ev, ppc, cost_detail


def balanced_partition(bdg_all: List[nx.DiGraph], vol_all: List[int], tool: str, dir_tool: str, params: Dict[str, Any]):
    """
        balanced graph partition on bdg
    """
    type_pkg = params["type_pkg"]
    w_power = params["w_power"]
    w_perf = params["w_perf"]
    w_cost = params["w_cost"]

    res_all = []
    for bdg in bdg_all:
        g_size = len(bdg)
        res_sys = []
        for k in range(1, g_size):
            if tool == "chaco":
                assert bdg.number_of_edges()
                ptt_sys = mincut_chaco(dir_chaco=dir_tool, G=bdg, k=k, clean=True)
            else:
                ptt_sys = mincut_metis(dir_metis=dir_tool, G=bdg, k=k, clean=True)
                ptt_sys = [p for p in ptt_sys if p]  # remove emtpy partition generated by metis
            cost_detail_sys = get_cost_detail(bdg_all=[bdg], vol_all=[1], type_pkg=type_pkg, ptt_all=[ptt_sys])
            RE_cost_keys = [
                "RE Cost of Raw Packages", "RE Cost of Package Defects", "RE Cost of Raw Dies", "RE Cost of Die Defects",
                "RE Cost of Wasted KGDs"
            ]
            RE_cost_sys = sum([cost_detail_sys[k][0] for k in RE_cost_keys])
            res_sys.append((RE_cost_sys, ptt_sys))
        res_all.append(min(res_sys, key=lambda e: e[0]))
    ptt_all = [ptt for _, ptt in res_all]
    ev, ppc = eval_ev_ppc(bdg_all=bdg_all,
                          vol_all=vol_all,
                          w_power=w_power,
                          w_perf=w_perf,
                          w_cost=w_cost,
                          type_pkg=type_pkg,
                          ptt_all=ptt_all)
    cost_detail = get_cost_detail(bdg_all=bdg_all, vol_all=vol_all, type_pkg=type_pkg, ptt_all=ptt_all)
    print(ptt_all)
    return ev, ppc, cost_detail


def get_cn_naive_single(bc_target: List[Block], bn_targets: List[Tuple[int, ...]], vol_all: List[int], params: Dict[str, Any],
                        cn_part: List[int], cn_remain: List[int]):
    res = []
    for cr in product(*cn_remain):
        cn_all = cn_part + cr
        pkgs = []
        for idx_sys, cn_sys in enumerate(cn_all):
            cpls: Dict[Chiplet, int] = {}
            blocks = {}
            for blk, num_blk in zip(bc_target, bn_targets[idx_sys]):
                blocks[blk] = num_blk // cn_sys
                assert num_blk % cn_sys == 0
            cpl = Chiplet(blocks=blocks)
            cpls[cpl] = cn_sys
            pkgs.append(make_package(type_pkg=params["type_pkg"], chiplets=cpls))
        cost_all = get_cost(pkgs=pkgs, vols=vol_all)
        res.append([sum(cost_all) / len(cost_all), cn_all])
    res_ = min(res, key=lambda e: e[0])
    return res_


@utils.timing
def get_cn_naive(bdg_all: List[nx.DiGraph], vol_all: List[int], params: Dict[str, Any]):
    """
        Test for adjust_cn, loop all possible combinations.
    """
    bd_all, _ = get_bd_bid(bdg_all=bdg_all)
    bc_target = list(set([blk for bd in bd_all for blk in bd]))

    bn_targets: List[Tuple[int, ...]] = []  # [(4, 4), (2, 2)]
    for idx_sys in range(len(bdg_all)):
        bn_targets.append(tuple(bd_all[idx_sys][blk] for blk in bc_target))

    num_proc = 64
    pool = Pool(processes=num_proc)
    res, res_ = [], []
    cns = utils.find_divisors_tuple(bn_targets)
    num_comb_part = 1
    for i in range(len(cns)):
        if num_comb_part * len(cns[i]) > num_proc:
            break
        num_comb_part *= len(cns[i])
    comb_part = list(product(*cns[:i]))
    for div_part in comb_part:
        res.append(pool.apply_async(get_cn_naive_single, args=(bc_target, bn_targets, vol_all, params, div_part, cns[i:])))
    pool.close()
    pool.join()
    for r in res:
        res_.append(r.get())
    print(min(res_, key=lambda e: e[0]))


BlockNumber = Tuple[int, ...]


@utils.timing
def get_cn_paper(bdg_all: List[nx.DiGraph], vol_all: List[int], params: Dict[str, Any]):
    """
        Algo presented in paper
    """
    bd_all, _ = get_bd_bid(bdg_all=bdg_all)
    bc_target = list(set([blk for bd in bd_all for blk in bd]))
    bn_all: List[Tuple[int, ...]] = []  # [(4, 4), (2, 2)]
    for idx_sys in range(len(bdg_all)):
        bn_all.append(tuple(bd_all[idx_sys][blk] for blk in bc_target))

    nbd = {}
    for bd_sys, vol_sys in zip(bd_all, vol_all):
        for blk in bd_sys:
            if blk not in nbd:
                nbd[blk] = 0
            nbd[blk] += bd_sys[blk] * vol_sys

    fac: Dict[int, Set[BlockNumber]] = {}  # {idx_bn_0: {(1, 2), {2, 4}, {4, 8}}}
    factors = utils.find_factors_tuple(bn_all)
    for idx_bn, f_l in enumerate(factors):
        fac[idx_bn] = set(f_l)

    reuse_graph = nx.Graph()
    reuse_graph.add_nodes_from(range(len(bdg_all)))
    for i in range(len(bdg_all)):
        for j in range(i + 1, len(bdg_all)):
            if fac[i] & fac[j]:
                reuse_graph.add_edge(i, j)

    eval_cache: Dict[Tuple[Set[int], BlockNumber], float] = {}  # key is ({idx_bn_0, idx_bn_1, ...}, block number)
    sol_cache: Dict[Set[int],
                    Tuple[float,
                          Dict[int,
                               BlockNumber]]] = {}  # key is {idx_bn_0, idx_bn_1, ...}, value is (cost, {idx_bn: block number})

    v, p = get_cn_paper_(bc_target=bc_target,
                         bns=bn_all,
                         targets=frozenset(range(len(bdg_all))),
                         eval_cache=eval_cache,
                         sol_cache=sol_cache,
                         fac=fac,
                         reuse_graph=reuse_graph,
                         vol_all=vol_all,
                         nbd=nbd,
                         params=params)
    cn_all = []
    for i in range(len(bdg_all)):
        cn_all.append(bn_all[i][0] // p[i][0])
    print(v / len(bdg_all), cn_all)


def get_cn_paper_(bc_target: List[Block], bns: List[BlockNumber], targets: Set[int],
                  eval_cache: Dict[Tuple[Set[int], BlockNumber],
                                   float], sol_cache: Dict[Set[int], float], fac: Dict[int, Set[BlockNumber]],
                  reuse_graph: nx.Graph, vol_all: List[int], nbd: Dict[Block, int], params: Dict[str, Any]):
    reuse_graph_targets = reuse_graph.subgraph(targets)
    O = 0
    P = {}
    for cc in nx.connected_components(reuse_graph_targets):
        cc = frozenset(cc)
        div = {}
        for i in cc:
            for f in fac[i]:
                if f not in div:
                    div[f] = []
                div[f].append(i)
        for f in div:
            div[f] = frozenset(div[f])

        res_f = []
        for f in div:
            p_f = {}
            if (div[f], f) in eval_cache:
                cost_this = eval_cache[(div[f], f)]
            else:
                pkgs = []
                for i in div[f]:
                    cpls: Dict[Chiplet, int] = {}
                    blocks = {}
                    for blk, num_blk in zip(bc_target, f):
                        blocks[blk] = num_blk
                    cpl = Chiplet(blocks=blocks)
                    cpls[cpl] = bns[i][0] // f[0]
                    pkgs.append(make_package(type_pkg=params["type_pkg"], chiplets=cpls))
                cost_this = sum(get_cost(pkgs=pkgs, vols=[vol_all[i] for i in div[f]], nbd=nbd))
                eval_cache[(div[f], f)] = cost_this
            for i in div[f]:
                p_f[i] = f

            if cc - div[f] in sol_cache:
                cost_sub, p_f_sub = sol_cache[cc - div[f]]
            else:
                cost_sub, p_f_sub = get_cn_paper_(bc_target=bc_target,
                                                  bns=bns,
                                                  targets=cc - div[f],
                                                  eval_cache=eval_cache,
                                                  sol_cache=sol_cache,
                                                  fac=fac,
                                                  reuse_graph=reuse_graph,
                                                  vol_all=vol_all,
                                                  nbd=nbd,
                                                  params=params)
                sol_cache[targets - div[f]] = (cost_sub, p_f_sub)
            p_f.update(p_f_sub)
            res_f.append(((cost_this + cost_sub), p_f))
        o, p = min(res_f)
        O += o
        P.update(p)
    return O, P


def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1, 2) (1, 3) (2, 3) (1, 2, 3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def reuse_first(bdg_all: List[nx.DiGraph], vol_all: List[int], dir_cplex: str, params: Dict[str, Any]):
    type_pkg = params["type_pkg"]
    w_power = params["w_power"]
    w_perf = params["w_perf"]
    w_cost = params["w_cost"]

    dir_tmp = tempfile.mkdtemp(dir=os.path.join(dir_cplex, "build"))
    path_file = os.path.join(dir_tmp, "solve.lp")

    M = 1e5
    bd_all, bid_all = get_bd_bid(bdg_all=bdg_all)

    idx_of_cpl = {}
    cpl_of_idx: List[List[Chiplet]] = []
    for idx_sys, bd_sys in enumerate(bd_all):
        bc_sys = list(powerset(bd_sys.keys()))  # chiplet candidate
        bn_sys_raw = [tuple(bd_sys[blk] for blk in bc) for bc in bc_sys]
        bn_sys = [tuple(n // math.gcd(*bn) for n in bn) for bn in bn_sys_raw]

        cpl_sys = []
        for idx_cpl, (bc, bn) in enumerate(zip(bc_sys, bn_sys)):
            cpl = Chiplet(blocks=dict(zip(bc, bn)), comm=None)
            if cpl not in idx_of_cpl:
                idx_of_cpl[cpl] = []
            idx_of_cpl[cpl].append((idx_sys, idx_cpl))
            cpl_sys.append(cpl)
        cpl_of_idx.append(cpl_sys)

    content = ""
    content += "Minimize {} z + s\n".format(M)
    content += "Subject To\n"
    for idx_sys, bd_sys in enumerate(bd_all):
        for blk in bd_sys:
            cstr_x = []
            for idx_cpl, cpl in enumerate(cpl_of_idx[idx_sys]):
                if blk in cpl.blocks:
                    cstr_x.append("x_{}_{}".format(idx_sys, idx_cpl))
            content += " + ".join(cstr_x) + " = 1\n"

    cpl_list: List[Chiplet] = list(idx_of_cpl)  # obtain fixed iterate order
    for idx_cpl, cpl in enumerate(cpl_list):
        indices = idx_of_cpl[cpl]
        assert indices
        if len(indices) == 1:
            content += "y_{} - x_{}_{} = 0\n".format(idx_cpl, indices[0][0], indices[0][1])
        else:
            for idx_sys, idx_cpl_same in indices:
                content += "y_{} - x_{}_{} >= 0\n".format(idx_cpl, idx_sys, idx_cpl_same)
            content += "y_{} - ".format(idx_cpl) + " - ".join(
                ["x_{}_{}".format(idx_sys, idx_cpl_same) for idx_sys, idx_cpl_same in indices]) + " <= 0\n"
    content += "z - " + " - ".join(["y_{}".format(idx_cpl) for idx_cpl in range(len(cpl_list))]) + " = 0\n"
    # area optimization
    content += "s - " + " - ".join(["{} y_{}".format(cpl.area_base, idx_cpl)
                                    for idx_cpl, cpl in enumerate(cpl_list)]) + " = 0\n"

    content += "Binary\n"
    content += " ".join(["y_{}".format(idx_bg) for idx_bg in range(len(cpl_list))]) + "\n"
    content += "End"

    with open(path_file, "w") as f:
        f.write(content)

    cpx = cplex.Cplex(path_file)
    cpx.set_log_stream(None)
    cpx.set_error_stream(None)
    cpx.set_warning_stream(None)
    cpx.set_results_stream(None)
    cpx.solve()

    bc_all, cn_all = [], []
    for idx_sys in range(len(bd_all)):
        bc_sys, cn_sys = [], []
        for idx_cpl in range(len(cpl_of_idx[idx_sys])):
            x = cpx.solution.get_values("x_{}_{}".format(idx_sys, idx_cpl))
            if x > 0:
                cpl = cpl_of_idx[idx_sys][idx_cpl]
                bc_cpl = list(cpl.blocks.keys())
                cn_cpl = bd_all[idx_sys][bc_cpl[0]] // cpl.blocks[bc_cpl[0]]
                bc_sys.append(set(bc_cpl))
                cn_sys.append(cn_cpl)
        bc_all.append(bc_sys)
        cn_all.append(cn_sys)

    shutil.rmtree(dir_tmp)

    ev, ppc = eval_ev_ppc(bdg_all=bdg_all,
                          vol_all=vol_all,
                          w_power=w_power,
                          w_perf=w_perf,
                          w_cost=w_cost,
                          type_pkg=type_pkg,
                          bc_all=bc_all,
                          cn_all=cn_all,
                          bid_all=bid_all)
    cost_detail = get_cost_detail(bdg_all=bdg_all,
                                  vol_all=vol_all,
                                  type_pkg=type_pkg,
                                  bc_all=bc_all,
                                  cn_all=cn_all,
                                  bid_all=bid_all)
    print(cn_all)
    return ev, ppc, cost_detail