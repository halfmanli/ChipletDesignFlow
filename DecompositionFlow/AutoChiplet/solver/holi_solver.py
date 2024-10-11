import math
import os
import random
from copy import deepcopy
from functools import reduce
from itertools import product
from typing import Any, Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from .. import utils
from ..model import (Block, Chiplet, get_cost, get_pp, graph_partition,
                     make_package)

BlockNumber = Tuple[int, ...]


class HoliSolver:
    """
        This SA solver is for chipletization of multiple soc instances.
    """
    def __init__(self, bdg_all: List[nx.DiGraph], vol_all: List[int], bc_all: List[List[Set[Block]]], cn_all: List[List[int]],
                 dir_log: str, params: Dict[str, Any]) -> None:
        """
            bdg_all: each item is the block description graph of SoC: vertex attribute "block" is Block object,
                and edge attribute "comm" is communication rate
            vol_all: each item is volume of the SoC
            bc_all: each item is the block combination of SoC
            cn_all: each item is the chiplet number of SoC(corresponding to block combination)
            ppc_all: we do not use this, because individual ppc do not consider reuse, making it incorrect; we recompute them instead
        """
        assert len(bdg_all) == len(vol_all) == len(bc_all) == len(cn_all)

        self.bdg_all = bdg_all
        self.vol_all = vol_all

        self.bd_all: List[Dict[Block, int]] = []  # each item is block description of the SoC: {"cpu":2, "gpu":4}
        self.bid_all: List[Dict[Block, List[int]]] = []  # block id/name dict of each system
        self.nbd: Dict[Block, int] = {}  # number of blocks in all systems
        for bdg, vol in zip(bdg_all, vol_all):
            bd = {}
            bid = {}
            for nid, attr in bdg.nodes(data=True):
                blk = attr["block"]
                if blk not in bd:
                    bd[blk] = 0
                    bid[blk] = []
                if blk not in self.nbd:
                    self.nbd[blk] = 0
                bd[blk] += 1
                bid[blk].append(nid)
                self.nbd[blk] += vol
            self.bd_all.append(bd)
            self.bid_all.append(bid)

        self.dir_log = dir_log
        if params is None:
            params = {}
        self.type_pkg = params["type_pkg"]
        self.max_try = params["holi_max_try"]
        self.num_init_sample = params["holi_num_init_sample"]
        self.T_start = params["holi_T_start"]
        self.T_end = params["holi_T_end"]
        self.alpha = params["holi_alpha"]
        self.w_power: float = params["w_power"]
        self.w_perf: float = params["w_perf"]
        self.w_cost: float = params["w_cost"]

        assert len(bc_all) == len(cn_all)
        for bc_sys, cn_sys in zip(bc_all, cn_all):  # [{cpu, gpu}, {ram}] [2, 4]
            assert len(bc_sys) == len(cn_sys)

        self.state_bc_cur: List[List[Set[Block]]] = bc_all  # block combination
        self.state_cn_cur: List[List[int]] = cn_all  # chiplet number
        self.ppc_cur: List[Tuple[float, float, float]] = None
        self.ev_cur: float = None
        self.ev_cur, self.ppc_cur = self.get_ev_ppc(state_bc=bc_all, state_cn=cn_all)  # recompute due to considering reuse

        self.state_bc_best: List[List[Set[Block]]] = self.state_bc_cur
        self.state_cn_best: List[List[int]] = self.state_cn_cur
        self.ppc_best = self.ppc_cur
        self.ev_best: float = self.ev_cur

    def get_ev_ppc(self, state_bc: List[List[Set[Block]]], state_cn: List[List[int]]):
        """
            Get energy value of state_bc&state_cn without optimization
        """
        assert len(state_bc) == len(state_cn) == len(self.bdg_all)
        power, perf = [], []
        for idx_sys, (bc_sys, cn_sys) in enumerate(zip(state_bc, state_cn)):  # evaluate all systems
            pm = np.zeros(shape=(len(self.bdg_all[idx_sys]), len(self.bdg_all[idx_sys])), dtype=int)
            assert len(bc_sys) == len(cn_sys)
            for bc, cn in zip(bc_sys, cn_sys):
                nid_bc = reduce(list.__add__, [self.bid_all[idx_sys][blk] for blk in bc])
                g_bc = nx.subgraph(self.bdg_all[idx_sys], nid_bc)
                ptt = graph_partition(g=g_bc, n=cn)
                for p in ptt:
                    for i in p:
                        for j in p:
                            pm[i][j] = 1  # same partition, same chiplet
            po, pe = get_pp(self.bdg_all[idx_sys], pm)
            power.append(po)
            perf.append(pe)

        pkgs = []
        for idx_sys in range(len(state_bc)):
            cpls: Dict[Chiplet, int] = {}
            for bc, cn in zip(state_bc[idx_sys], state_cn[idx_sys]):
                blocks = {}
                for blk in bc:
                    blocks[blk] = self.bd_all[idx_sys][blk] // cn
                    assert self.bd_all[idx_sys][blk] % cn == 0
                cpl = Chiplet(blocks=blocks)  # ignore d2d area
                cpls[cpl] = cn
            pkgs.append(make_package(type_pkg=self.type_pkg, chiplets=cpls))
        cost = get_cost(pkgs=pkgs, vols=self.vol_all, nbd=self.nbd)

        ev = self.w_power * np.average(power) + self.w_perf * np.average(perf) + self.w_cost * np.average(cost)
        ppc = list(zip(power, perf, cost))
        return ev, ppc

    def get_ev(self, ncd_other: Dict[Chiplet, int], bc_target: Set[Block], idx_sys_targets: List[int],
               idx_bc_targets: List[int], state_bc: List[List[Set[Block]]], state_cn: List[List[int]],
               state_pm_fixed: List[np.ndarray], pp_cache: Dict[Tuple[int, int], Tuple[float, float]]):
        power, perf = [], []
        for idx_sys, idx_bc in zip(idx_sys_targets, idx_bc_targets):
            assert bc_target == state_bc[idx_sys][idx_bc]
            cn = state_cn[idx_sys][idx_bc]
            if (idx_sys, cn) not in pp_cache:
                state_pm = state_pm_fixed[idx_sys].copy()
                nid_bc = reduce(list.__add__, [self.bid_all[idx_sys][blk] for blk in bc_target])
                g_bc = nx.subgraph(self.bdg_all[idx_sys], nid_bc)
                ptt = graph_partition(g=g_bc, n=cn)
                for p in ptt:  # [[0, 1, 2], [3, 4]]
                    for i in p:
                        for j in p:
                            state_pm[i][j] = 1
                po, pe = get_pp(self.bdg_all[idx_sys], state_pm)
                pp_cache[(idx_sys, cn)] = (po, pe, idx_bc)  # idx_bc is just for debug
            else:
                po, pe, idx_bc_ = pp_cache[(idx_sys, cn)]
                assert idx_bc == idx_bc_
            power.append(po)
            perf.append(pe)

        pkgs = []
        assert len(state_bc) == len(state_cn)
        ncd = deepcopy(ncd_other)
        for idx_sys in idx_sys_targets:  # only consider target system
            cpls: Dict[Chiplet, int] = {}
            assert len(state_bc[idx_sys]) == len(state_cn[idx_sys])
            for bc, cn in zip(state_bc[idx_sys], state_cn[idx_sys]):
                blocks = {}
                for blk in bc:
                    blocks[blk] = self.bd_all[idx_sys][blk] // cn
                    assert self.bd_all[idx_sys][blk] % cn == 0
                cpl = Chiplet(blocks=blocks)
                cpls[cpl] = cn
                if bc == bc_target:
                    if cpl not in ncd:
                        ncd[cpl] = 0
                    ncd[cpl] += self.vol_all[idx_sys] * cn
            pkgs.append(make_package(type_pkg=self.type_pkg, chiplets=cpls))
        cost = get_cost(pkgs=pkgs, vols=[self.vol_all[idx_sys] for idx_sys in idx_sys_targets], nbd=self.nbd, ncd=ncd)

        ev_sum = sum(self.w_power * np.array(power)) + sum(self.w_perf * np.array(perf)) + sum(self.w_cost * np.array(cost))
        indic = {}
        for idx in range(len(idx_sys_targets)):
            indic[idx_sys_targets[idx]] = (power[idx], perf[idx], cost[idx])
        return ev_sum, indic

    def get_cn(self, bns: List[BlockNumber], bc_target: Set[Block], idx_sys_bns: List[int], idx_bc_bns: List[int],
               state_bc: List[List[Set[Block]]], state_cn: List[List[int]]):
        """
            bns: block number list, [(4, 8), (9, 6)], all block numbe tuples have the same resource order
            idx_sys_bns[i], idx_bc_bns[i]: the sys/bc index of bn_targets[i]
        """
        assert len(bns) == len(idx_sys_bns) == len(idx_bc_bns)

        # prepare unchanged parts of state_pm
        state_pm_fixed: List[np.ndarray] = []
        for idx_sys in range(len(state_bc)):
            if idx_sys not in idx_sys_bns:
                state_pm_fixed.append(None)  # systems will not be evaluated
            else:
                idx_bc_t = idx_bc_bns[idx_sys_bns.index(idx_sys)]  # find the corresponding idx_bc_t
                bdg = self.bdg_all[idx_sys]
                pm_fixed = np.zeros(shape=(len(bdg), len(bdg)), dtype=int)
                for idx_bc in range(len(state_bc[idx_sys])):
                    if idx_bc != idx_bc_t:  # partition matrix of fixed parts
                        nid_bc = reduce(list.__add__, [self.bid_all[idx_sys][blk] for blk in state_bc[idx_sys][idx_bc]])
                        g_bc = nx.subgraph(bdg, nid_bc)
                        ptt = graph_partition(g=g_bc, n=state_cn[idx_sys][idx_bc])
                        for p in ptt:  # [[0, 1, 2], [3, 4]]
                            for i in p:
                                for j in p:
                                    pm_fixed[i][j] = 1
                state_pm_fixed.append(pm_fixed)
        assert len(state_bc) == len(state_cn) == len(state_pm_fixed)

        eval_cache: Dict[Tuple[Set[int], BlockNumber], float] = {}  # key is ({idx_bn_0, idx_bn_1, ...}, block number)
        sol_cache: Dict[Set[int], Tuple[float, Dict[int, BlockNumber], Dict[int, Tuple[float, float, float]]]] = {
        }  # key is {idx_bn_0, idx_bn_1, ...}, value is (cost, {idx_bn: block number}, indicator)
        pp_cache: Dict[Tuple[int, int], Tuple[float,
                                              float]] = {}  # power, perf of (idx_sys, chiplet number) for specific bc_target

        # divisible by block number; {(4, 8): {idx_bn_0, idx_bn_1}, (2, 4): {idx_bn_0, idx_bn_1}}
        # note the index is not idx_sys, idx_bn instead
        div: Dict[BlockNumber, Set[int]] = {}
        indiv: Dict[BlockNumber, Set[int]] = {}  # {(4, 8): {idx_bn_2}, (2, 4): {idx_bn_2}}
        fac: Dict[int, Set[BlockNumber]] = {}  # {idx_bn_0: {(1, 2), {2, 4}, {4, 8}}}
        factors = utils.find_factors_tuple(bns)
        div_tmp: Dict[BlockNumber, List[int]] = {}

        ncd_other: Dict[Chiplet, int] = {}  # number of chipelt dict
        for idx_sys in range(len(state_bc)):
            for bc, cn in zip(state_bc[idx_sys], state_cn[idx_sys]):
                if bc == bc_target:
                    continue
                blocks = {}
                for blk in bc:
                    blocks[blk] = self.bd_all[idx_sys][blk] // cn
                    assert self.bd_all[idx_sys][blk] % cn == 0
                cpl = Chiplet(blocks=blocks)
                if cpl not in ncd_other:
                    ncd_other[cpl] = 0
                ncd_other[cpl] += cn * self.vol_all[idx_sys]

        for idx_bn, f_l in enumerate(factors):
            fac[idx_bn] = set(f_l)
            for f in f_l:
                if f not in div_tmp:
                    div_tmp[f] = []
                div_tmp[f].append(idx_bn)
        for f in div_tmp.keys():
            div[f] = frozenset(div_tmp[f])
            indiv[f] = set(range(len(bns))) - set(div[f])
        _, p, indic_new = self.get_cn_(bns=bns,
                                       targets=frozenset(range(len(bns))),
                                       bc_target=bc_target,
                                       idx_sys_bns=idx_sys_bns,
                                       idx_bc_bns=idx_bc_bns,
                                       state_bc=state_bc,
                                       state_cn=state_cn,
                                       ncd_other=ncd_other,
                                       eval_cache=eval_cache,
                                       sol_cache=sol_cache,
                                       div=div,
                                       indiv=indiv,
                                       fac=fac,
                                       state_pm_fixed=state_pm_fixed,
                                       pp_cache=pp_cache)
        state_cn_new = deepcopy(state_cn)
        for idx_bn, (bn, idx_sys, idx_bc) in enumerate(zip(bns, idx_sys_bns, idx_bc_bns)):
            state_cn_new[idx_sys][idx_bc] = bn[0] // p[idx_bn][0]
        return state_cn_new, indic_new

    def get_cn_(self, bns: List[BlockNumber], targets: Set[int], bc_target: Set[Block], idx_sys_bns: List[int],
                idx_bc_bns: List[int], state_bc: List[List[Set[Block]]], state_cn: List[List[int]], ncd_other: Dict[Chiplet,
                                                                                                                    int],
                eval_cache: Dict[Tuple[Set[int], BlockNumber],
                                 Dict[str, Any]], sol_cache: Dict[Set[int], Tuple[float, Dict[int, BlockNumber],
                                                                                  Dict[int, Tuple[float, float, float]]]],
                div: Dict[BlockNumber, Set[int]], indiv: Dict[BlockNumber, Set[int]], fac: Dict[int, Set[BlockNumber]],
                state_pm_fixed: List[np.ndarray], pp_cache: Dict[Tuple[int, int], Tuple[float, float]]):
        """
            targets: idx of remained bn to be optimized 
        """
        assert targets  # should not be empty
        res_all = []
        for bn_t in div:  # bn tuple
            div_targets = div[bn_t] & targets  # this is index
            indiv_targets = indiv[bn_t] & targets
            if not div_targets:
                continue
            res_v: Dict[int, float] = 0  # energy value of targets
            res_p: Dict[int, BlockNumber] = {}  # result of partition
            res_indic: Dict[int, Tuple[float, float, float]] = {}  # result of indicators
            # greedy selection
            for idx_bn in div_targets:
                res_p[idx_bn] = bn_t
            # evaluating div_targets
            e_k = (div_targets, bn_t)  # key for eval_cache
            if e_k in eval_cache:
                res_v_d, res_indic_d = eval_cache[e_k]
            else:
                state_cn_comb = deepcopy(state_cn)
                for idx_bn in div_targets:
                    state_cn_comb[idx_sys_bns[idx_bn]][idx_bc_bns[idx_bn]] = bns[idx_bn][0] // bn_t[0]
                res_v_d, res_indic_d = self.get_ev(ncd_other=ncd_other,
                                                   bc_target=bc_target,
                                                   idx_sys_targets=[idx_sys_bns[idx_bn] for idx_bn in div_targets],
                                                   idx_bc_targets=[idx_bc_bns[idx_bn] for idx_bn in div_targets],
                                                   state_bc=state_bc,
                                                   state_cn=state_cn_comb,
                                                   state_pm_fixed=state_pm_fixed,
                                                   pp_cache=pp_cache)
                eval_cache[e_k] = (res_v_d, res_indic_d)
            res_indic.update(res_indic_d)
            res_v += res_v_d

            if indiv_targets:
                to_sel = list(indiv_targets)
                num_to_sel = len(to_sel)
                reuse_graph = nx.Graph()
                reuse_graph.add_nodes_from(to_sel)
                for i in range(num_to_sel):
                    for j in range(i + 1, num_to_sel):
                        idx_soc_i, idx_soc_j = to_sel[i], to_sel[j]
                        if fac[idx_soc_i] & fac[idx_soc_j]:
                            reuse_graph.add_edge(idx_soc_i, idx_soc_j)

                for comp in nx.connected_components(reuse_graph):
                    target_sub = frozenset(comp)  # subproblem
                    if target_sub in sol_cache:
                        res_v_sub, res_p_sub, res_indic_sub = sol_cache[target_sub]
                    else:
                        res_v_sub, res_p_sub, res_indic_sub = self.get_cn_(bns=bns,
                                                                           targets=target_sub,
                                                                           bc_target=bc_target,
                                                                           idx_sys_bns=idx_sys_bns,
                                                                           idx_bc_bns=idx_bc_bns,
                                                                           state_bc=state_bc,
                                                                           state_cn=state_cn,
                                                                           ncd_other=ncd_other,
                                                                           eval_cache=eval_cache,
                                                                           sol_cache=sol_cache,
                                                                           div=div,
                                                                           indiv=indiv,
                                                                           fac=fac,
                                                                           state_pm_fixed=state_pm_fixed,
                                                                           pp_cache=pp_cache)
                    res_v += res_v_sub
                    res_p.update(res_p_sub)
                    res_indic.update(res_indic_sub)
            res_all.append([res_v, res_p, res_indic])

        if targets not in sol_cache:
            sol_cache[targets] = min(res_all, key=lambda e: e[0])
        return min(res_all, key=lambda e: e[0])

    def adjust_cn(self,
                  bc_target: Set[Block],
                  state_bc: List[List[Set[Block]]],
                  state_cn: List[List[int]],
                  ppc: List[Tuple[float, float, float]] = None) -> Tuple[float, List[List[int]]]:
        """
            For all bc_target in state_bc, return the optimal chiplet number.
            Pass state_bc, state_cn instead of using self.state_bc, self.state_cn directly for using init_sample
            Return: energy value, optimized state_cn, new state_pm
        """
        idx_sys_targets, idx_bc_targets = [], []  # idx of system containing bc; idx of target bc
        for idx_sys in range(len(state_bc)):
            try:
                idx_bc = state_bc[idx_sys].index(bc_target)
            except ValueError:
                continue
            else:
                idx_sys_targets.append(idx_sys)
                idx_bc_targets.append(idx_bc)
        assert idx_sys_targets  # should not be empty

        bc_target_list = sorted(bc_target)  # to ensure block number stable
        bn_targets: List[BlockNumber] = []  # [(4, 4), (2, 2)]
        for idx_sys in idx_sys_targets:
            bn_targets.append(tuple(self.bd_all[idx_sys][blk] for blk in bc_target_list))
        state_cn_neigh, indic_neigh = self.get_cn(bns=bn_targets,
                                                  bc_target=bc_target,
                                                  idx_sys_bns=idx_sys_targets,
                                                  idx_bc_bns=idx_bc_targets,
                                                  state_bc=state_bc,
                                                  state_cn=state_cn)
        ev_neigh, ppc_neigh = self.get_ev_ppc(state_bc=state_bc, state_cn=state_cn_neigh)
        return ev_neigh, state_cn_neigh, ppc_neigh

    def apply_reuse_bc2sys(self, bc_target: Set[Block], state_bc: List[List[Set[Block]]], state_cn: List[List[int]],
                           ppc: List[Tuple[float, float, float]]):
        """
            Apply the block combination/partition (state_bc[idx_sys]) to all other system
        """
        state_bc_new, state_cn_new = [], []
        ppc_new = ppc
        bc_adjust = [bc_target]  # bc needs to adjust chiplet number
        flg_bc_target = False  # bc_target exists in at least one state_bc
        for idx_sys in range(len(state_bc)):
            if not set().union(*state_bc[idx_sys]).issuperset(bc_target):
                state_bc_sys = [set(bc) for bc in state_bc[idx_sys]]
                state_cn_sys = state_cn[idx_sys][:]
            else:
                flg_bc_target = True
                state_bc_sys, state_cn_sys = [], []
                for bc, cn in zip(state_bc[idx_sys], state_cn[idx_sys]):
                    if bc & bc_target:
                        bc = bc - bc_target
                        if bc:  # bc = bc_target in some cases
                            state_bc_sys.append(bc)
                            state_cn_sys.append(cn)  # chiplet number unchanged
                            bc_adjust.append(bc)
                    else:
                        state_bc_sys.append(set(bc))  # shallow copy
                        state_cn_sys.append(cn)
                state_bc_sys.append(set(bc_target))
                state_cn_sys.append([-1])  # error if bugs exist
            state_bc_new.append(state_bc_sys)
            state_cn_new.append(state_cn_sys)
        assert flg_bc_target
        for bc in bc_adjust:
            ev, state_cn_new, ppc_new = self.adjust_cn(bc_target=bc, state_bc=state_bc_new, state_cn=state_cn_new, ppc=ppc_new)

        return ev, (state_bc_new, state_cn_new, ppc_new)

    def apply_reuse_sys2sys(self, idx_sys: int, state_bc: List[List[Set[Block]]], state_cn: List[List[int]],
                            ppc: List[Tuple[float, float, float]]):
        """
            Apply the block combination/partition (state_bc[idx_sys]) to all other system
        """
        state_bc_tmp, state_cn_tmp, ppc_tmp = state_bc, state_cn, ppc
        for bc in state_bc[idx_sys]:
            ev, (state_bc_tmp, state_cn_tmp, ppc_tmp) = self.apply_reuse_bc2sys(bc_target=bc,
                                                                                state_bc=state_bc_tmp,
                                                                                state_cn=state_cn_tmp,
                                                                                ppc=ppc_tmp)
        return ev, (state_bc_tmp, state_cn_tmp, ppc_tmp)

    def random_choice_2D(self, population: List[List[Any]], weights: List[List[float]]):
        """
            Select an element from 2d weight list according to the weight
        """
        weights_list = reduce(lambda x, y: x + y, weights)
        population_list = reduce(lambda x, y: x + y, population)
        return random.choices(population=population_list, weights=weights_list, k=1)[0]

    def can_share(self, l_a: List[int], l_b: List[int]):
        gcd_a = utils.find_gcd(l_a)
        gcd_b = utils.find_gcd(l_b)
        l_a_base = [e // gcd_a for e in l_a]
        l_b_base = [e // gcd_b for e in l_b]
        return l_a_base == l_b_base

    def get_reuse_score(self, state_bc: List[List[Set[Block]]]):
        """
            reuse_score[idx_sys][idx_bc]: the occurrence number of state_bc[idx_sys][idx_bc] in other system
        """
        assert len(self.bd_all) == len(state_bc)
        reuse_score = []
        for idx_sys_i in range(len(state_bc)):
            reuse_score_sys = []
            for idx_bc_i in range(len(state_bc[idx_sys_i])):
                reuse_score_bc = 0
                blks = list(state_bc[idx_sys_i][idx_bc_i])
                bn_i = [self.bd_all[idx_sys_i][b] for b in blks]
                for idx_sys_j in range(len(state_bc)):
                    if idx_sys_j == idx_sys_i:
                        continue
                    # decide whether state_bc[idx_sys_i][idx_bc_i] can be reused in state_bc[idx_sys_i]
                    bn_j = [self.bd_all[idx_sys_j][b] if (b in self.bd_all[idx_sys_j]) else -1 for b in blks]
                    if -1 in bn_j:
                        continue
                    elif self.can_share(bn_i, bn_j):
                        reuse_score_bc += 1 * len(blks)  # prefer larger reuse block combination
                reuse_score_sys.append(reuse_score_bc)
            reuse_score.append(reuse_score_sys)
        return reuse_score

    def neighbor(self,
                 weight_act: List[float],
                 state_bc: List[List[Set[Block]]] = None,
                 state_cn: List[List[int]] = None,
                 ppc: List[Tuple[float, float, float]] = None):
        """
            ev_neigh: energy value of neighbor solution
        """
        if state_bc is None:
            assert state_cn is None and ppc is None
            state_bc, state_cn, ppc = self.state_bc_cur, self.state_cn_cur, self.ppc_cur
        assert len(state_bc) == len(state_cn) == len(ppc)

        MOVE = 0  # move one type of block to another combination
        SPLIT = 1  # split two combination
        MERGE = 2  # merge two combination
        SWAP = 3  # swap two types of blocks in two combinations
        APPLY_REUSE_BC = 4
        APPLY_REUSE_SYS = 5

        act = random.choices(population=[MOVE, SPLIT, MERGE, SWAP, APPLY_REUSE_BC, APPLY_REUSE_SYS], k=1, weights=weight_act)[0]
        for _ in range(self.max_try):
            if act == MOVE:
                idx_sys = random.randint(0, len(state_bc) - 1)
                if len(state_bc[idx_sys]) == 1:
                    continue
                state_bc_all_neigh = deepcopy(state_bc)
                state_cn_all_neigh = deepcopy(state_cn)
                state_bc_neigh = state_bc_all_neigh[idx_sys]
                state_cn_neigh = state_cn_all_neigh[idx_sys]
                idx_src_comb, idx_dst_comb = random.sample(range(len(state_bc_neigh)), k=2)
                src_comb, dst_comb = state_bc_neigh[idx_src_comb], state_bc_neigh[idx_dst_comb]
                blk = random.choice(tuple(src_comb))
                if len(src_comb) == 1:
                    dst_comb.add(blk)
                    state_bc_neigh.pop(idx_src_comb)  # delete empty set
                    state_cn_neigh.pop(idx_src_comb)
                    ev_neigh, state_cn_all_neigh, ppc_neigh = self.adjust_cn(bc_target=dst_comb,
                                                                             state_bc=state_bc_all_neigh,
                                                                             state_cn=state_cn_all_neigh,
                                                                             ppc=ppc)
                else:
                    src_comb.remove(blk)
                    dst_comb.add(blk)
                    state_cn_neigh[idx_dst_comb] = 1  # change to 1 for block added
                    _, state_cn_all_neigh, ppc_neigh = self.adjust_cn(bc_target=src_comb,
                                                                      state_bc=state_bc_all_neigh,
                                                                      state_cn=state_cn_all_neigh,
                                                                      ppc=ppc)
                    ev_neigh, state_cn_all_neigh, ppc_neigh = self.adjust_cn(bc_target=dst_comb,
                                                                             state_bc=state_bc_all_neigh,
                                                                             state_cn=state_cn_all_neigh,
                                                                             ppc=ppc_neigh)
                return ev_neigh, (state_bc_all_neigh, state_cn_all_neigh, ppc_neigh)

            elif act == SPLIT:
                idx_sys_i = random.randint(0, len(state_bc) - 1)
                state_bc_all_neigh = deepcopy(state_bc)
                state_cn_all_neigh = deepcopy(state_cn)
                ppc_neigh = deepcopy(ppc)
                state_bc_neigh = state_bc_all_neigh[idx_sys_i]
                state_cn_neigh = state_cn_all_neigh[idx_sys_i]
                # delete original block combination
                idx_comb = random.randint(0, len(state_bc_neigh) - 1)
                comb = state_bc_neigh.pop(idx_comb)
                if len(comb) < 2:
                    continue
                cn = state_cn_neigh.pop(idx_comb)
                comb = list(comb)
                random.shuffle(comb)
                reuse_cnt = 1
                new_comb_reuse = set()
                new_comb_nonreuse = set()

                for b_comb in comb:
                    comb_try = new_comb_reuse | {b_comb}
                    reuse_cnt_cur = 0
                    blks = list(comb_try)
                    bn_i = [self.bd_all[idx_sys_i][b] for b in blks]
                    for idx_sys_j in range(len(state_bc)):
                        if idx_sys_i == idx_sys_j:
                            continue
                        bn_j = [self.bd_all[idx_sys_j][b] if (b in self.bd_all[idx_sys_j]) else -1 for b in blks]
                        if -1 in bn_j:
                            continue
                        elif self.can_share(bn_i, bn_j):
                            reuse_cnt_cur += 1
                    if reuse_cnt_cur >= reuse_cnt:
                        new_comb_reuse |= {b_comb}
                        reuse_cnt = reuse_cnt_cur
                    else:
                        new_comb_nonreuse |= {b_comb}

                if new_comb_reuse:
                    state_bc_neigh += [new_comb_reuse]
                    state_cn_neigh += [cn]
                if new_comb_nonreuse:
                    state_bc_neigh += [new_comb_nonreuse]
                    state_cn_neigh += [cn]

                if new_comb_reuse:
                    ev_neigh, state_cn_all_neigh, ppc_neigh = self.adjust_cn(bc_target=new_comb_reuse,
                                                                             state_bc=state_bc_all_neigh,
                                                                             state_cn=state_cn_all_neigh,
                                                                             ppc=ppc_neigh)
                if new_comb_nonreuse:
                    ev_neigh, state_cn_all_neigh, ppc_neigh = self.adjust_cn(bc_target=new_comb_nonreuse,
                                                                             state_bc=state_bc_all_neigh,
                                                                             state_cn=state_cn_all_neigh,
                                                                             ppc=ppc_neigh)

                return ev_neigh, (state_bc_all_neigh, state_cn_all_neigh, ppc_neigh)

            elif act == MERGE:
                idx_sys = random.randint(0, len(state_bc) - 1)
                if len(state_bc[idx_sys]) == 1:
                    continue
                state_bc_all_neigh = deepcopy(state_bc)
                state_cn_all_neigh = deepcopy(state_cn)
                state_bc_neigh = state_bc_all_neigh[idx_sys]
                state_cn_neigh = state_cn_all_neigh[idx_sys]
                # delete original block combinations
                idx_comb_1 = random.randint(0, len(state_bc_neigh) - 1)
                comb_1 = state_bc_neigh.pop(idx_comb_1)
                state_cn_neigh.pop(idx_comb_1)
                idx_comb_2 = random.randint(0, len(state_bc_neigh) - 1)
                comb_2 = state_bc_neigh.pop(idx_comb_2)
                state_cn_neigh.pop(idx_comb_2)
                # merge two block combinations
                comb_new = comb_1 | comb_2
                state_bc_neigh.append(comb_new)
                state_cn_neigh.append(0)
                ev_neigh, state_cn_all_neigh, ppc_neigh = self.adjust_cn(bc_target=comb_new,
                                                                         state_bc=state_bc_all_neigh,
                                                                         state_cn=state_cn_all_neigh,
                                                                         ppc=ppc)
                return ev_neigh, (state_bc_all_neigh, state_cn_all_neigh, ppc_neigh)

            elif act == SWAP:
                idx_sys = random.randint(0, len(state_bc) - 1)
                if len(state_bc[idx_sys]) == 1:
                    continue
                state_bc_all_neigh = deepcopy(state_bc)
                state_cn_all_neigh = deepcopy(state_cn)
                state_bc_neigh = state_bc_all_neigh[idx_sys]
                state_cn_neigh = state_cn_all_neigh[idx_sys]
                idx_comb_1, idx_comb_2 = random.sample(range(len(state_bc_neigh)), k=2)
                comb_1, comb_2 = state_bc_neigh[idx_comb_1], state_bc_neigh[idx_comb_2]
                blk_1, blk_2 = random.choice(tuple(comb_1)), random.choice(tuple(comb_2))
                comb_1.remove(blk_1)
                comb_1.add(blk_2)
                comb_2.remove(blk_2)
                comb_2.add(blk_1)
                state_cn_neigh[idx_comb_1] = 1  # 1 is always correct
                state_cn_neigh[idx_comb_2] = 1
                _, state_cn_neigh, ppc_neigh = self.adjust_cn(bc_target=comb_1,
                                                              state_bc=state_bc_all_neigh,
                                                              state_cn=state_cn_all_neigh,
                                                              ppc=ppc)
                ev_neigh, state_cn_neigh, ppc_neigh = self.adjust_cn(bc_target=comb_2,
                                                                     state_bc=state_bc_all_neigh,
                                                                     state_cn=state_cn_all_neigh,
                                                                     ppc=ppc_neigh)
                return ev_neigh, (state_bc_all_neigh, state_cn_all_neigh, ppc_neigh)

            elif act == APPLY_REUSE_BC:
                reuse_score = self.get_reuse_score(state_bc=state_bc)
                if max([max(rs_sys) for rs_sys in reuse_score]) <= 0.0:
                    return None
                comb = self.random_choice_2D(population=state_bc, weights=reuse_score)
                return self.apply_reuse_bc2sys(bc_target=comb, state_bc=state_bc, state_cn=state_cn, ppc=ppc)

            elif act == APPLY_REUSE_SYS:
                reuse_score_bc = self.get_reuse_score(state_bc=state_bc)
                reuse_score_sys = [sum(rs_sys) for rs_sys in reuse_score_bc]
                if max(reuse_score_sys) <= 0.0:
                    return None
                else:
                    idx_sys = random.choices(population=list(range(len(state_bc))), weights=reuse_score_sys, k=1)[0]
                    return self.apply_reuse_sys2sys(idx_sys=idx_sys, state_bc=state_bc, state_cn=state_cn, ppc=ppc)

            else:
                assert False
        return None

    def init_sample(self, num_sample):
        """
            Get the average for normalization
        """
        ev_sample = []
        state_bc, state_cn, ppc = self.state_bc_cur, self.state_cn_cur, self.ppc_cur
        i = 0
        while i < num_sample:
            sol_neigh = self.neighbor(weight_act=[1] * 6, state_bc=state_bc, state_cn=state_cn,
                                      ppc=ppc)  # all weight with the same possibility
            if sol_neigh is None:
                continue
            else:
                ev, (state_bc, state_cn, ppc) = sol_neigh
                ev_sample.append(ev)
                i += 1
        self.ev_norm = sum(ev_sample) / len(ev_sample)

    def log(self, plot=False):
        """
            plot: plot & save image if True
        """
        indic_name = ["power", "perf", "cost"]
        if self.hist is None:
            self.hist = {}
            for i_n in indic_name:
                self.hist[i_n] = [[] for _ in range(len(self.bdg_all))]
            self.hist["ev"] = []

        for idx_sys in range(len(self.bdg_all)):
            power_sys, perf_sys, cost_sys = self.ppc_cur[idx_sys]
            self.hist["power"][idx_sys].append(power_sys)
            self.hist["perf"][idx_sys].append(perf_sys)
            self.hist["cost"][idx_sys].append(cost_sys)
        self.hist["ev"].append(self.ev_cur)

        if plot:
            _, axes = plt.subplots(2, 2, figsize=(20, 20))
            for idx_i_n, i_n in enumerate(indic_name):
                for idx_sys in range(len(self.bdg_all)):
                    ax = sns.lineplot(ax=axes[idx_i_n // 2][idx_i_n % 2], data=self.hist[i_n][idx_sys], label=str(idx_sys))
                ax.set_title(i_n)
            sns.lineplot(ax=axes[1][1], data=self.hist["ev"]).set_title("ev")
            plt.savefig(os.path.join(self.dir_log, "log_detailed.png"))
            plt.close()

            _, axes = plt.subplots(2, 2, figsize=(20, 20))
            for idx_i_n, i_n in enumerate(indic_name):
                ax = sns.lineplot(ax=axes[idx_i_n // 2][idx_i_n % 2], data=np.average(np.array(self.hist[i_n]), axis=0))
                ax.set_title(i_n)
            sns.lineplot(ax=axes[1][1], data=self.hist["ev"]).set_title("ev")
            plt.savefig(os.path.join(self.dir_log, "log.png"))
            plt.close()
            with open(os.path.join(self.dir_log, "best_sol.txt"), "w") as f:
                f.write(str(self.ev_best) + "\n")
                f.write(str(self.ppc_best)+ "\n")
                f.write(str(self.state_bc_best) + "\n")
                f.write(str(self.state_cn_best) + "\n")

    def opt(self):
        """
            SA optimization process.
        """
        self.init_sample(self.num_init_sample)
        T = self.T_start
        self.hist = None
        while T > self.T_end:
            for cnt_iloop in range(24):  # loop for same temperature
                # MOVE, SPLIT, MERGE, SWAP, APPLY_REUSE_BC, APPLY_REUSE_SYS
                if cnt_iloop % 6 < 4:
                    weight_act = [2, 1, 1, 2, 0, 0]  # explore
                else:
                    weight_act = [0, 0, 0, 0, 4, 1]  # exploit

                sol_neigh = self.neighbor(weight_act=weight_act)
                if sol_neigh is not None:
                    ev_neigh, (state_bc_neigh, state_cn_neigh, ppc_neigh) = sol_neigh
                    delta_ev = self.ev_cur - ev_neigh
                    ap = math.exp(delta_ev / self.ev_norm / T)
                    if random.random() <= ap:  # accept neighboring solution
                        self.ev_cur = ev_neigh
                        self.state_bc_cur = state_bc_neigh
                        self.state_cn_cur = state_cn_neigh
                        self.ppc_cur = ppc_neigh
                        if ev_neigh < self.ev_best:
                            self.ev_best = ev_neigh
                            self.state_bc_best = state_bc_neigh
                            self.state_cn_best = state_cn_neigh
                            self.ppc_best = ppc_neigh
                    self.log(plot=(cnt_iloop % 20 == 0))
            T *= self.alpha
        return self.ev_best, (self.state_bc_best, self.state_cn_best, self.ppc_best)