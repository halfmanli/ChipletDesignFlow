import os
from multiprocessing import Pool
import networkx as nx
from typing import List, Dict, Set, Any
from .indi_solver import IndiSolver
from .holi_solver import HoliSolver
from ..model import Block, get_cost_detail, get_bd_bid, eval_ev_ppc


def indi_opt(bdg: nx.DiGraph, vol: int, dir_log: str, params: Dict[str, Any]):
    isolver = IndiSolver(bdg=bdg, vol=vol, dir_log=dir_log, params=params)
    return isolver.opt()


def holi_opt(bdg_all: List[nx.DiGraph], vol_all: List[int], bc_all: List[List[Set[Block]]], cn_all: List[List[int]],
             dir_log: str, params: Dict[str, Any]):
    hsolver = HoliSolver(bdg_all=bdg_all, vol_all=vol_all, bc_all=bc_all, cn_all=cn_all, dir_log=dir_log, params=params)
    return hsolver.opt()


def chiplet_partition(bdg_all: List[nx.DiGraph], vol_all: List[int], params: Dict[str, Any]):
    """
        task_name: for logging the data
    """
    num_cpu = params["num_cpu"]  # number of process in pool
    dir_log_root = params["dir_log_root"]  # dir of log
    indi_pnum = params["indi_pnum"]  # number of optimization process per bdg
    holi_pnum = params["holi_pnum"]  # number of opitmization of holistic optimization

    # individual optimization
    pool = Pool(processes=num_cpu)
    res_wait_indi = []
    dir_log_indi = os.path.join(dir_log_root, "indi")
    os.mkdir(dir_log_indi)
    for idx_sys, (bdg, vol) in enumerate(zip(bdg_all, vol_all)):
        dir_log_indi_sys = os.path.join(dir_log_indi, str(idx_sys))
        os.mkdir(dir_log_indi_sys)
        res_wait_indi_sys = []
        for pid in range(indi_pnum):
            dir_log_indi_p = os.path.join(dir_log_indi_sys, str(pid))  # log of each optimization process
            os.mkdir(dir_log_indi_p)
            res_wait_indi_sys.append(pool.apply_async(indi_opt, args=(bdg, vol, dir_log_indi_p, params)))
        res_wait_indi.append(res_wait_indi_sys)
    pool.close()
    pool.join()
    bc_all, cn_all = [], []
    for idx_sys in range(len(res_wait_indi)):
        res_indi_sys = []
        for pid in range(indi_pnum):
            res_indi_sys.append(res_wait_indi[idx_sys][pid].get())
        _, (bc_sys, cn_sys) = min(res_indi_sys, key=lambda e: e[0])
        bc_all.append(bc_sys)
        cn_all.append(cn_sys)

    # holistic optimization
    pool = Pool(processes=num_cpu)
    res_wait_holi = []
    dir_log_holi = os.path.join(dir_log_root, "holi")
    os.mkdir(dir_log_holi)
    for pid in range(holi_pnum):
        dir_log_holi_p = os.path.join(dir_log_holi, str(pid))  # log of each optimization process
        os.mkdir(dir_log_holi_p)
        res_wait_holi.append(pool.apply_async(holi_opt, args=(bdg_all, vol_all, bc_all, cn_all, dir_log_holi_p, params)))
    pool.close()
    pool.join()
    res_holi = []
    for pid in range(holi_pnum):
        res_holi.append(res_wait_holi[pid].get())
    _, (state_bc_best, state_cn_best, _) = (min(res_holi, key=lambda e: e[0]))

    _, bid_all = get_bd_bid(bdg_all=bdg_all)
    ev, ppc = eval_ev_ppc(bdg_all=bdg_all,
                          vol_all=vol_all,
                          bc_all=state_bc_best,
                          cn_all=state_cn_best,
                          bid_all=bid_all,
                          w_power=params["w_power"],
                          w_perf=params["w_perf"],
                          w_cost=params["w_cost"],
                          type_pkg=params["type_pkg"])
    cost_detail = get_cost_detail(bdg_all=bdg_all,
                                  vol_all=vol_all,
                                  type_pkg=params["type_pkg"],
                                  bc_all=state_bc_best,
                                  cn_all=state_cn_best,
                                  bid_all=bid_all)
    return ev, ppc, cost_detail