import pickle
from datetime import datetime
from os import makedirs
from os import path as osp

import numpy as np

from .. import dataset
from ..solver import chiplet_partition


def tip(s):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("{}: {}".format(s, dt_string))


dir_root = osp.join(osp.dirname(osp.abspath(__file__)), "../..")
dir_metis = dir_chaco = osp.join(dir_root, "tool/metis")
dir_cplex = dir_chaco = osp.join(dir_root, "tool/cplex")

for mode in range(2):
    """
        0: remove separate/individual optimization
        1: remove holistic optimization
    """
    for vol_single in [500 * 1000, 10 * 1000 * 1000]:
        ppc = {}  # overall ppc
        cd = {}  # cost details
        for strategy in ["CP", "M", "RF", "BP", "FG", "C"]:
            ppc[strategy] = {"power": [], "perf": [], "cost": []}
            cd[strategy] = []
        dir_log_root = osp.join(dir_root, "log/exp_2/%d/%d" % (mode, vol_single))
        ds = dataset.get_dataset()
        for ds_name in ["AMD", "HiSilicon", "Intel", "Nvidia", "Rockchip"]:
            makedirs(osp.join(dir_log_root, ds_name))
            if mode == 0:
                params = {
                    "num_cpu": 32,
                    "indi_pnum": 2,
                    "indi_max_try": 10,
                    "indi_T_start": 1,
                    "indi_T_end": 100,
                    "indi_alpha": 0.95,
                    "holi_pnum": 16,
                    "holi_num_init_sample": 20,
                    "holi_max_try": 10,
                    "holi_T_start": 1,
                    "holi_T_end": 0.05,
                    "holi_alpha": 0.95,
                    "dir_log_root": osp.join(dir_log_root, ds_name),
                    "type_pkg": "SI",
                    "w_power": 0,
                    "w_perf": 0,
                    "w_cost": 1
                }
            else:
                params = {
                    "num_cpu": 32,
                    "indi_pnum": 2,
                    "indi_max_try": 10,
                    "indi_T_start": 1,
                    "indi_T_end": 0.1,
                    "indi_alpha": 0.95,
                    "holi_pnum": 16,
                    "holi_num_init_sample": 20,
                    "holi_max_try": 10,
                    "holi_T_start": 1,
                    "holi_T_end": 100,
                    "holi_alpha": 0.95,
                    "dir_log_root": osp.join(dir_log_root, ds_name),
                    "type_pkg": "SI",
                    "w_power": 0,
                    "w_perf": 0,
                    "w_cost": 1
                }

            bdg_all = ds[ds_name]
            vol_all = [vol_single] * len(bdg_all)
            ev_cp, ppc_cp, cost_detail_cp = chiplet_partition(bdg_all=bdg_all, vol_all=vol_all, params=params)

            power_cp, perf_cp, cost_cp = zip(*ppc_cp)

            ppc["CP"]["power"].append(power_cp)
            ppc["CP"]["perf"].append(perf_cp)
            ppc["CP"]["cost"].append(cost_cp)

            cd["CP"].append(cost_detail_cp)

        res = [ppc, cd]
        with open(osp.join(dir_log_root, "res.pickle"), "wb") as f:
            pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
