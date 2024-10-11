import pickle
from datetime import datetime
from os import makedirs
from os import path as osp

import numpy as np

from .. import baseline, dataset
from ..solver import chiplet_partition


def tip(s):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("{}: {}".format(s, dt_string))


def get_data():
    dir_root = osp.join(osp.dirname(osp.abspath(__file__)), "../..")
    dir_metis = dir_chaco = osp.join(dir_root, "tool/metis")
    dir_cplex = dir_chaco = osp.join(dir_root, "tool/cplex")

    vol_single = 500 * 1000
    vol_single = 10 * 1000 * 1000
    ppc = {}  # overall ppc
    cd = {}  # cost details
    for strategy in ["CP", "M", "RF", "BP", "FG", "C"]:
        ppc[strategy] = {"power": [], "perf": [], "cost": []}
        cd[strategy] = []
    dir_log_root = osp.join(dir_root, "log/exp_cn/%d" % vol_single)
    tip("BEG")
    ds = dataset.get_dataset()
    for ds_name in ["AMD", "HiSilicon", "Intel", "Nvidia", "Rockchip"]:
        if ds_name != "Nvidia":
            indi_T_end, holi_T_end = 0.2, 0.5
        else:
            indi_T_end, holi_T_end = 0.1, 0.1
        # makedirs(osp.join(dir_log_root, ds_name))
        params = {
            "num_cpu": 32,
            "indi_pnum": 2,
            "indi_max_try": 10,
            "indi_T_start": 1,
            "indi_T_end": indi_T_end,
            "indi_alpha": 0.95,
            "holi_pnum": 24,
            "holi_num_init_sample": 20,
            "holi_max_try": 10,
            "holi_T_start": 1,
            "holi_T_end": holi_T_end,
            "holi_alpha": 0.95,
            "dir_log_root": osp.join(dir_log_root, ds_name),
            "type_pkg": "SI",
            "w_power": 0,
            "w_perf": 0,
            "w_cost": 1
        }
        bdg_all = ds[ds_name]
        vol_all = [vol_single] * len(bdg_all)
        # ev_cp, ppc_cp, cost_detail_cp = chiplet_partition(bdg_all=bdg_all, vol_all=vol_all, params=params)
        # ev_m, ppc_m, cost_detail_m = baseline.monolithic(bdg_all=bdg_all, vol_all=vol_all, params=params)
        # ev_rf, ppc_rf, cost_detail_rf = baseline.reuse_first(bdg_all=bdg_all,
        #                                                      vol_all=vol_all,
        #                                                      dir_cplex=dir_cplex,
        #                                                      params=params)
        # ev_bp, ppc_bp, cost_detail_bp = baseline.balanced_partition(bdg_all=bdg_all,
        #                                                             vol_all=vol_all,
        #                                                             tool="metis",
        #                                                             dir_tool=dir_metis,
        #                                                             params=params)
        ev_fg, ppc_fg, cost_detail_fg = baseline.finest_granularity(bdg_all=bdg_all, vol_all=vol_all, params=params)
        # ev_c, ppc_c, cost_detail_c = baseline.chopin(bdg_all=bdg_all, vol_all=vol_all, x=1.5, params=params)

def rf():
    dir_root = osp.join(osp.dirname(osp.abspath(__file__)), "../..")
    dir_metis = dir_chaco = osp.join(dir_root, "tool/metis")
    dir_cplex = dir_chaco = osp.join(dir_root, "tool/cplex")

    vol_single = 500 * 1000
    vol_single = 10 * 1000 * 1000
    ppc = {}  # overall ppc
    cd = {}  # cost details
    for strategy in ["CP", "M", "RF", "BP", "FG", "C"]:
        ppc[strategy] = {"power": [], "perf": [], "cost": []}
        cd[strategy] = []
    dir_log_root = osp.join(dir_root, "log/exp_1/%d" % vol_single)
    tip("BEG")
    ds = dataset.get_dataset()
    for ds_name in ["AMD", "HiSilicon", "Intel", "Nvidia", "Rockchip"]:
        if ds_name != "Rockchip":
            continue
        if ds_name != "Nvidia":
            indi_T_end, holi_T_end = 0.2, 0.5
        else:
            indi_T_end, holi_T_end = 0.1, 0.1
        params = {
            "num_cpu": 32,
            "indi_pnum": 2,
            "indi_max_try": 10,
            "indi_T_start": 1,
            "indi_T_end": indi_T_end,
            "indi_alpha": 0.95,
            "holi_pnum": 24,
            "holi_num_init_sample": 20,
            "holi_max_try": 10,
            "holi_T_start": 1,
            "holi_T_end": holi_T_end,
            "holi_alpha": 0.95,
            "dir_log_root": osp.join(dir_log_root, ds_name),
            "type_pkg": "SI",
            "w_power": 0,
            "w_perf": 0,
            "w_cost": 1
        }
        bdg_all = ds[ds_name]
        vol_all = [vol_single] * len(bdg_all)
        ev_m, ppc_m, cost_detail_m = baseline.monolithic(bdg_all=bdg_all, vol_all=vol_all, params=params)
        ev_rf, ppc_rf, cost_detail_rf = baseline.reuse_first(bdg_all=bdg_all,
                                                             vol_all=vol_all,
                                                             dir_cplex=dir_cplex,
                                                             params=params)


def sum_(l):
    return sum([sum(ll) for ll in l])

def compare():
    num_cp_500k = sum_([[2, 4], [4, 4], [6, 4], [8, 4], [1, 1], [2, 1]]) + sum_([[1, 8], [4, 1], [1], [4, 1], [
        1, 8
    ], [1, 4]]) + sum_([[10, 2, 3], [19, 2, 3], [8, 2, 3], [9, 2, 3], [7, 2, 3], [5, 2, 2], [4, 2, 2], [5, 2, 2], [3, 2, 2],
                        [2, 2, 2]]) + sum_([[2, 8, 1], [14, 6, 1], [10, 1, 1], [8, 4, 1], [5, 3, 1], [4, 2, 1]]) + sum_(
                            [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]])

    num_rf_500k = sum_([[4, 4], [8, 4], [12, 4], [16, 4], [2, 1], [4, 1]]) + sum_([[16, 1], [8, 1], [1], [1, 8], [32, 1], [
        1, 8
    ]]) + sum_([[40, 3, 4], [38, 3, 4], [32, 3, 4], [36, 3, 4], [28, 3, 4], [20, 2, 4], [16, 2, 4], [10, 2, 4], [12, 2, 4],
                [8, 2, 4]]) + sum_([[128, 40, 1, 2], [84, 6, 1, 12], [60, 5, 1, 8], [48, 4, 1, 8], [30, 3, 1, 6], [24, 2, 1, 4]
                                    ]) + sum_([[1], [1], [1], [1], [1], [1], [1], [1, 1], [1], [1]])

    print(num_rf_500k / num_cp_500k)

    num_fg_500k = sum_([[16, 4, 4, 8], [32, 8, 4, 8], [48, 12, 4, 8], [64, 16, 4, 8], [8, 2, 1, 2], [16, 4, 1, 2]]) + sum_([[
        16, 16, 1, 1, 1, 1, 1, 1
    ], [8, 8, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [8, 8, 1], [32, 1, 1, 1, 1, 1, 1], [8, 8, 1]]) + sum_([
        [40, 4, 3, 4], [38, 4, 3, 4], [32, 4, 3, 4], [36, 4, 3, 4], [28, 4, 3, 4], [20, 4, 2, 4], [16, 4, 2, 4], [10, 4, 2, 4],
        [12, 4, 2, 4], [8, 4, 2, 4]
    ]) + sum_([[128, 40, 1, 6, 4], [84, 6, 1, 12], [60, 5, 1, 8], [48, 4, 1, 8], [30, 3, 1, 6], [24, 2, 1, 4]]) + sum_(
        [[4, 2, 1, 1, 1, 1, 1], [4, 2, 1, 1, 1, 1, 1], [4, 2, 1, 1, 1, 1, 1], [4, 2, 4, 2, 1, 2, 1], [4, 2, 4, 3, 1, 1, 2, 1],
         [4, 4, 2, 1, 1, 1], [8, 1, 1, 1, 1, 1], [4, 2, 1, 1, 1, 1], [4, 2, 1, 1, 1], [4, 1, 1]])

    print(num_fg_500k / num_cp_500k)

    num_cp_10M = sum_([[2, 1], [2, 2], [2, 3], [2, 4], [1, 2], [1, 1]]) + sum_(
        [[4, 1], [2, 1], [1], [1, 2], [4, 1], [1, 2]]) + sum_([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]) + sum_(
            [[4, 1], [1], [1], [1], [1], [1]]) + sum_([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]])
    
    print("Chipletizer comparision:", num_cp_10M / num_cp_500k)
    print("Chipletizer comparision of monolithic:", 26 / 11)


if __name__ == "__main__":
    # get_data()
    compare()
