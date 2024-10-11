from copy import deepcopy
from functools import reduce
from itertools import accumulate, product
import math
import os
from typing import List, Dict, Set, Any, Tuple
from ..model import Block, Chiplet, make_package, graph_partition, get_cost, get_pp
import networkx as nx
from .. import utils
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class IndiSolver:
    """
        Individual SA solver for chipletization of each soc instance.
    """
    def __init__(self, bdg: nx.DiGraph, vol: int, dir_log: str, params: Dict[str, Any]) -> None:
        """
            bdg: block description graph of the SoC: vertex attribute "block" is Block object,
                and edge attribute "comm" is communication rate
            vol: manufacturing volume of the SoC
            dir_log: the dir to output log
        """
        assert set(bdg) == set(range(len(bdg)))  # node name of bdg should be 0 ~ len(bdg) - 1
        assert nx.number_of_selfloops(bdg) == 0

        self.bdg = bdg
        self.vol = vol
        self.dir_log = dir_log
        self.max_try = params["indi_max_try"]
        self.type_pkg = params["type_pkg"]
        self.T_start = params["indi_T_start"]
        self.T_end = params["indi_T_end"]
        self.alpha = params["indi_alpha"]
        self.w_power = params["w_power"]
        self.w_perf = params["w_perf"]
        self.w_cost = params["w_cost"]

        # extract block description (bd) from bdg_all
        self.bd: Dict[Block, int] = {}  # block description of the SoC: {cpu: 2, gpu: 4}
        self.bid: Dict[Block, List[int]] = {}  # block-node id dict {cpu: [0, 1], gpu:[2, 3, 4, 5]}
        for nid, attr in bdg.nodes(data=True):
            blk = attr["block"]
            if blk not in self.bd:
                self.bd[blk] = 0
                self.bid[blk] = []
            self.bd[blk] += 1
            self.bid[blk].append(nid)
        self.blks = list(self.bd.keys())

        self.state_bc_cur: List[Set[Block]] = None  # block combination, [{cpu, gpu}, {ddr}]
        self.state_cn_cur: List[int] = None  # chiplet number
        self.state_pm_cur: np.ndarray = None  # partition matrix, state_pm[i][j] is 1 if block i, j in the same chiplet
        self.ppc_cur: Tuple[float, float, float] = None  #  current power, perf, cost
        self.ev_cur: float = None  # current energy value

        self.state_bc_best: List[Set[Block]] = None  # best historical solution
        self.state_cn_best: List[int] = None
        self.state_pm_best: np.ndarray = None
        self.ppc_best: Tuple[float, float, float] = None
        self.ev_best: float = None

    def get_ev(self, targets: List[int], state_bc: List[Set[Block]], state_cn: List[int],
               state_pm_fixed: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
            Return: energy value, indicator dict
            targets: indices of target bc
        """
        state_pm = state_pm_fixed.copy()
        for idx_bc in range(len(state_bc)):
            if idx_bc in targets:  # partition matrix of fixed parts
                nid_bc = reduce(list.__add__, [self.bid[blk] for blk in state_bc[idx_bc]])
                g_bc = nx.subgraph(self.bdg, nid_bc)
                ptt = graph_partition(g=g_bc, n=state_cn[idx_bc])
                for p in ptt:  # [[0, 1, 2], [3, 4]]
                    for i in p:
                        for j in p:
                            state_pm[i][j] = 1
        power, perf = get_pp(self.bdg, state_pm)

        cpls: Dict[Chiplet, int] = {}
        for bc, cn in zip(state_bc, state_cn):
            blocks = {}
            for blk in bc:
                blocks[blk] = self.bd[blk] // cn
                assert self.bd[blk] % cn == 0
            cpl = Chiplet(blocks=blocks)
            cpls[cpl] = cn
        pkgs = [make_package(type_pkg=self.type_pkg, chiplets=cpls)]
        cost = get_cost(pkgs=pkgs, vols=[self.vol])[0]

        ev = self.w_power * power + self.w_perf * perf + self.w_cost * cost

        return ev, {"power": power, "perf": perf, "cost": cost, "state_pm": state_pm}

    def get_cn(self, targets: List[int], state_bc: List[Set[Block]], state_cn: List[int]):
        """
            Get the optimal number of chiplets, return state_cn_new
            state_bc: block combination
            state_cn: chiplet numbers, only targets will be considered
            targets: indices of considered bc
        """
        bn_targets = []  # [(4, 4), (2,)]
        for idx_bc in targets:
            bc = state_bc[idx_bc]  # block combination: (cpu, gpu)
            bn_targets.append(tuple(self.bd[blk] for blk in bc))

        state_pm_fixed = np.zeros(shape=(len(self.bdg), len(self.bdg)), dtype=int)
        for idx_bc in range(len(state_bc)):
            if idx_bc not in targets:  # partition matrix of fixed parts
                nid_bc = reduce(list.__add__, [self.bid[blk] for blk in state_bc[idx_bc]])
                g_bc = nx.subgraph(self.bdg, nid_bc)
                ptt = graph_partition(g=g_bc, n=state_cn[idx_bc])
                for p in ptt:  # [[0, 1, 2], [3, 4]]
                    for i in p:
                        for j in p:
                            state_pm_fixed[i][j] = 1

        # candidate chiplet number of each block combination
        cn_cands = utils.find_divisors_tuple(bn_targets)
        num_cn_cands = reduce((lambda x, y: x * y), map(len, cn_cands))
        if num_cn_cands < 1000:
            cn_combs = list(product(*cn_cands))  # [[1, 2, 4], [1, 2]] -> [(1, 1), (1, 2), (2, 1), (2, 2) ...]
            res_combs = []
            for cn_comb in cn_combs:  # (1, 1)
                state_cn_comb = state_cn[:]
                for idx_cn, cn in enumerate(cn_comb):
                    state_cn_comb[targets[idx_cn]] = cn
                ev, indic = self.get_ev(targets=targets,
                                        state_bc=state_bc,
                                        state_cn=state_cn_comb,
                                        state_pm_fixed=state_pm_fixed)
                res_combs.append((ev, state_cn_comb, indic))
        else:
            assert False
        return min(res_combs, key=lambda e: e[0])

    def init_opt(self, method="random"):
        """
            Generate initial state. 
        """
        if method not in ["random"]:
            raise ValueError("Error: invalid method.")

        if method == "random":
            idx_split = [0] + list(accumulate(utils.split_num(num=len(self.blks))))
            blks = self.blks[:]
            random.shuffle(blks)
            state_bc = [set(blks[beg:end]) for (beg, end) in zip(idx_split, idx_split[1:])]
            ev, state_cn, indic = self.get_cn(targets=list(range(len(state_bc))),
                                              state_bc=state_bc,
                                              state_cn=[0] *
                                              len(state_bc))

            self.state_bc_cur = state_bc
            self.state_cn_cur = state_cn
            self.state_pm_cur = indic["state_pm"]
            self.ppc_cur = (indic["power"], indic["perf"], indic["cost"])
            self.ev_cur = ev

            self.state_bc_best = self.state_bc_cur
            self.state_cn_best = self.state_cn_cur
            self.state_pm_best = self.state_pm_cur
            self.ppc_best = self.ppc_cur
            self.ev_best = ev

    def init_sample(self, num_sample):
        """
            Get the average for normalization
        """
        ev_sample = []
        state_bc, state_cn = self.state_bc_cur, self.state_cn_cur
        i = 0
        while i < num_sample:
            sol_neigh = self.neighbor(state_bc=state_bc, state_cn=state_cn)
            if sol_neigh is None:
                continue
            else:
                ev, (state_bc, state_cn, _) = sol_neigh
                ev_sample.append(ev)
                i += 1
        self.ev_norm = sum(ev_sample) / len(ev_sample)

    def neighbor(self, state_bc: List[Set[Block]] = None, state_cn: List[int] = None):
        """
            ev_neigh: energy value of neighbor solution
        """
        if state_bc is None:
            assert state_cn is None
            state_bc = self.state_bc_cur
            state_cn = self.state_cn_cur
        assert len(state_bc) == len(state_cn)

        MOVE = 0  # move one type of block to another combination
        SPLIT = 1  # split two combination
        MERGE = 2  # merge two combination
        SWAP = 3  # swap two types of blocks in two combinations

        weight_act = [0.4, 0.1, 0.1, 0.4]
        act = random.choices([MOVE, SPLIT, MERGE, SWAP], k=1, weights=weight_act)[0]

        for _ in range(self.max_try):
            if act == MOVE:
                if len(state_bc) == 1:
                    continue
                state_bc_neigh = deepcopy(state_bc)
                state_cn_neigh_ = state_cn[:]
                idx_src_comb, idx_dst_comb = random.sample(range(len(state_bc_neigh)), k=2)
                src_comb, dst_comb = state_bc_neigh[idx_src_comb], state_bc_neigh[idx_dst_comb]
                blk = random.choice(tuple(src_comb))
                if len(src_comb) == 1:
                    dst_comb.add(blk)
                    state_bc_neigh.pop(idx_src_comb)  # delete empty set
                    state_cn_neigh_.pop(idx_src_comb)
                    ev_neigh, state_cn_neigh, indic_neigh = self.get_cn(
                        targets=[idx_dst_comb if idx_dst_comb < idx_src_comb else (idx_dst_comb - 1)],
                        state_bc=state_bc_neigh,
                        state_cn=state_cn_neigh_)
                else:
                    src_comb.remove(blk)
                    dst_comb.add(blk)
                    ev_neigh, state_cn_neigh, indic_neigh = self.get_cn(targets=[idx_src_comb, idx_dst_comb],
                                                                        state_bc=state_bc_neigh,
                                                                        state_cn=state_cn_neigh_)
                return ev_neigh, (state_bc_neigh, state_cn_neigh, indic_neigh)

            elif act == SPLIT:  # TODO: split according to process technology
                state_bc_neigh = deepcopy(state_bc)
                state_cn_neigh_ = state_cn[:]
                # delete original block combination
                idx_comb = random.randint(0, len(state_bc_neigh) - 1)
                comb = state_bc_neigh.pop(idx_comb)
                if len(comb) < 2:
                    continue
                state_cn_neigh_.pop(idx_comb)
                # evenly split
                comb = list(comb)
                random.shuffle(comb)
                new_comb_1 = set(comb[:len(comb) // 2])
                new_comb_2 = set(comb[len(comb) // 2:])
                state_bc_neigh += [new_comb_1, new_comb_2]
                state_cn_neigh_ += [0, 0]
                idx_new_comb_1 = len(state_cn_neigh_) - 1
                idx_new_comb_2 = len(state_cn_neigh_) - 2
                ev_neigh, state_cn_neigh, indic_neigh = self.get_cn(targets=[idx_new_comb_1, idx_new_comb_2],
                                                                    state_bc=state_bc_neigh,
                                                                    state_cn=state_cn_neigh_)
                return ev_neigh, (state_bc_neigh, state_cn_neigh, indic_neigh)

            elif act == MERGE:
                if len(state_bc) == 1:
                    continue
                state_bc_neigh = deepcopy(state_bc)
                state_cn_neigh_ = state_cn[:]
                # delete original block combinations
                idx_comb_1 = random.randint(0, len(state_bc_neigh) - 1)
                comb_1 = state_bc_neigh.pop(idx_comb_1)
                state_cn_neigh_.pop(idx_comb_1)
                idx_comb_2 = random.randint(0, len(state_bc_neigh) - 1)
                comb_2 = state_bc_neigh.pop(idx_comb_2)
                state_cn_neigh_.pop(idx_comb_2)
                # merge two block combinations
                state_bc_neigh.append(comb_1 | comb_2)
                state_cn_neigh_.append(0)
                idx_comb_new = len(state_bc_neigh) - 1
                ev_neigh, state_cn_neigh, indic_neigh = self.get_cn(targets=[idx_comb_new],
                                                                    state_bc=state_bc_neigh,
                                                                    state_cn=state_cn_neigh_)
                return ev_neigh, (state_bc_neigh, state_cn_neigh, indic_neigh)

            elif act == SWAP:
                if len(state_bc) == 1:
                    continue
                state_bc_neigh = deepcopy(state_bc)
                idx_comb_1, idx_comb_2 = random.sample(range(len(state_bc_neigh)), k=2)
                comb_1, comb_2 = state_bc_neigh[idx_comb_1], state_bc_neigh[idx_comb_2]
                blk_1, blk_2 = random.choice(tuple(comb_1)), random.choice(tuple(comb_2))
                comb_1.remove(blk_1)
                comb_1.add(blk_2)
                comb_2.remove(blk_2)
                comb_2.add(blk_1)
                ev_neigh, state_cn_neigh, indic_neigh = self.get_cn(targets=[idx_comb_1, idx_comb_2],
                                                                    state_bc=state_bc_neigh,
                                                                    state_cn=state_cn)
                return ev_neigh, (state_bc_neigh, state_cn_neigh, indic_neigh)
            else:
                assert False
        return None  # failed to get neighoring solution

    def opt(self):
        """
            SA optimization process.
        """
        self.init_opt()
        self.init_sample(20)
        hist = {"power": [], "perf": [], "cost": [], "ev": []}
        T = self.T_start
        while T > self.T_end:
            for cnt_iloop in range(len(self.blks)):
                sol_neigh = self.neighbor()
                if sol_neigh is None:
                    continue
                ev_neigh, (state_bc_neigh, state_cn_neigh, indic_neigh) = sol_neigh
                delta_ev = self.ev_cur - ev_neigh
                ap = math.exp(delta_ev / self.ev_norm / T)
                if random.random() <= ap:  # accept neighboring solution
                    self.ev_cur = ev_neigh
                    self.state_bc_cur = state_bc_neigh
                    self.state_cn_cur = state_cn_neigh
                    self.state_pm_cur = indic_neigh["state_pm"]
                    self.ppc_cur = (indic_neigh["power"], indic_neigh["perf"], indic_neigh["cost"])
                    if ev_neigh < self.ev_best:
                        self.ev_best = ev_neigh
                        self.state_bc_best = state_bc_neigh
                        self.state_cn_best = state_cn_neigh
                        self.state_pm_best = indic_neigh["state_pm"]
                        self.ppc_best = (indic_neigh["power"], indic_neigh["perf"], indic_neigh["cost"])
                # log
                hist["power"].append(self.ppc_cur[0])
                hist["perf"].append(self.ppc_cur[1])
                hist["cost"].append(self.ppc_cur[2])
                hist["ev"].append(self.ev_cur)
                if cnt_iloop % 10 == 0:
                    _, axes = plt.subplots(2, 2, figsize=(10, 10))
                    for idx_key, key in enumerate(hist):
                        sns.lineplot(ax=axes[idx_key // 2][idx_key % 2], data=hist[key], label=key)
                    plt.savefig(os.path.join(self.dir_log, "log.png"))
                    plt.close()
                    with open(os.path.join(self.dir_log, "best_sol.txt"), "w") as f:
                        f.write(str(self.ev_best) + "\n")
                        f.write(str(self.ppc_best) + "\n")
                        f.write(str(self.state_bc_best) + "\n")
                        f.write(str(self.state_cn_best) + "\n")
            T *= self.alpha
        return self.ev_best, (self.state_bc_best, self.state_cn_best)