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


dir_root = osp.join(osp.dirname(osp.abspath(__file__)), "../..")
dir_metis = dir_chaco = osp.join(dir_root, "tool/metis")
dir_cplex = dir_chaco = osp.join(dir_root, "tool/cplex")

vol_single = 500 * 1000
# vol_single = 10 * 1000 * 1000
ppc = {}  # overall ppc
cd = {}  # cost details
for strategy in ["CP", "M", "RF", "BP", "FG", "C"]:
    ppc[strategy] = {"power": [], "perf": [], "cost": []}
    cd[strategy] = []
dir_log_root = osp.join(dir_root, "log/exp_1_changechopin/%d" % vol_single)
tip("BEG")
ds = dataset.get_dataset()
for ds_name in ["AMD", "HiSilicon", "Intel", "Nvidia", "Rockchip"]:
    if ds_name != "Nvidia":
        indi_T_end, holi_T_end = 0.2, 0.5
    else:
        indi_T_end, holi_T_end = 0.1, 0.1
    makedirs(osp.join(dir_log_root, ds_name))
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
    ev_cp, ppc_cp, cost_detail_cp = chiplet_partition(bdg_all=bdg_all, vol_all=vol_all, params=params)
    ev_m, ppc_m, cost_detail_m = baseline.monolithic(bdg_all=bdg_all, vol_all=vol_all, params=params)
    ev_rf, ppc_rf, cost_detail_rf = baseline.reuse_first(bdg_all=bdg_all, vol_all=vol_all, dir_cplex=dir_cplex, params=params)
    ev_bp, ppc_bp, cost_detail_bp = baseline.balanced_partition(bdg_all=bdg_all,
                                                                vol_all=vol_all,
                                                                tool="metis",
                                                                dir_tool=dir_metis,
                                                                params=params)
    ev_fg, ppc_fg, cost_detail_fg = baseline.finest_granularity(bdg_all=bdg_all, vol_all=vol_all, params=params)
    ev_c, ppc_c, cost_detail_c = baseline.chopin(bdg_all=bdg_all, vol_all=vol_all, x=1.1, params=params)

    power_cp, perf_cp, cost_cp = zip(*ppc_cp)
    power_m, perf_m, cost_m = zip(*ppc_m)
    power_rf, perf_rf, cost_rf = zip(*ppc_rf)
    power_bp, perf_bp, cost_bp = zip(*ppc_bp)
    power_fg, perf_fg, cost_fg = zip(*ppc_fg)
    power_c, perf_c, cost_c = zip(*ppc_c)

    ppc["CP"]["power"].append(power_cp)
    ppc["CP"]["perf"].append(perf_cp)
    ppc["CP"]["cost"].append(cost_cp)
    ppc["M"]["power"].append(power_m)
    ppc["M"]["perf"].append(perf_m)
    ppc["M"]["cost"].append(cost_m)
    ppc["RF"]["power"].append(power_rf)
    ppc["RF"]["perf"].append(perf_rf)
    ppc["RF"]["cost"].append(cost_rf)
    ppc["BP"]["power"].append(power_bp)
    ppc["BP"]["perf"].append(perf_bp)
    ppc["BP"]["cost"].append(cost_bp)
    ppc["FG"]["power"].append(power_fg)
    ppc["FG"]["perf"].append(perf_fg)
    ppc["FG"]["cost"].append(cost_fg)
    ppc["C"]["power"].append(power_c)
    ppc["C"]["perf"].append(perf_c)
    ppc["C"]["cost"].append(cost_c)

    cd["CP"].append(cost_detail_cp)
    cd["M"].append(cost_detail_m)
    cd["RF"].append(cost_detail_rf)
    cd["BP"].append(cost_detail_bp)
    cd["FG"].append(cost_detail_fg)
    cd["C"].append(cost_detail_c)
    print(ds_name, ppc)

res = [ppc, cd]
with open(osp.join(dir_log_root, "res.pickle"), "wb") as f:
    pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
tip("END")