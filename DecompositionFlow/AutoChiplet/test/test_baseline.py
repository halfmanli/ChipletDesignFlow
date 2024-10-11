import os
from .. import baseline
from .. import dataset
from os import path as osp
import networkx as nx
from ..model import Block
from ..solver import HoliSolver

dir_root = osp.join(osp.dirname(osp.abspath(__file__)), "../..")
dir_chaco = osp.join(dir_root, "tool/chaco")
dir_metis = dir_chaco = osp.join(dir_root, "tool/metis")
dir_cplex = dir_chaco = osp.join(dir_root, "tool/cplex")


def test_rf():
    params = {"type_pkg": "SI", "w_power": 1, "w_perf": 1, "w_cost": 1}
    bdgs_intel = dataset.AMD()
    vol_intel = 500 * 1000
    print(baseline.reuse_first(bdg_all=bdgs_intel, vol_all=[vol_intel] * len(bdgs_intel), dir_cplex=dir_cplex, params=params))
    bdgs_hisilicon = dataset.HiSilicon()
    vol_hisilicon = 500 * 1000
    print(
        baseline.reuse_first(bdg_all=bdgs_hisilicon,
                             vol_all=[vol_hisilicon] * len(bdgs_hisilicon),
                             dir_cplex=dir_cplex,
                             params=params))


def test_mono():
    params = {"type_pkg": "SI", "w_power": 0, "w_perf": 0, "w_cost": 1}

    bdgs_intel = dataset.Intel()
    vol_intel = 20 * 1000 * 1000
    print(baseline.monolithic(bdg_all=bdgs_intel, vol_all=[vol_intel] * len(bdgs_intel), params=params))

    bdgs_hisilicon = dataset.HiSilicon()
    vol_hisilicon = 20 * 1000 * 1000
    print(baseline.monolithic(bdg_all=bdgs_hisilicon, vol_all=[vol_hisilicon] * len(bdgs_hisilicon), params=params))


def test_fg():
    params = {"type_pkg": "SI", "w_power": 0, "w_perf": 0, "w_cost": 1}

    bdgs_intel = dataset.Intel()
    vol_intel = 20 * 1000 * 1000
    print(baseline.finest_granularity(bdg_all=bdgs_intel, vol_all=[vol_intel] * len(bdgs_intel), params=params))

    bdgs_hisilicon = dataset.HiSilicon()
    vol_hisilicon = 20 * 1000 * 1000
    print(baseline.finest_granularity(bdg_all=bdgs_hisilicon, vol_all=[vol_hisilicon] * len(bdgs_hisilicon), params=params))


def test_in():
    params = {
        "num_cpu": 32,
        "pnum": 2,
        "dir_log_root": osp.join(dir_root, "log/test"),
        "indi_max_try": 10,
        "type_pkg": "SI",
        "indi_T_start": 1,
        "indi_T_end": 0.01,
        "indi_alpha": 0.95,
        "w_power": 1,
        "w_perf": 1,
        "w_cost": 1
    }
    bdgs_intel = dataset.Intel()
    vol_intel = 500 * 1000
    print(baseline.indp(bdg_all=bdgs_intel, vol_all=[vol_intel] * len(bdgs_intel), params=params))
    # bdgs_hisilicon = dataset.HiSilicon()
    # vol_intel = 500 * 1000
    # print(baseline.indp(bdg_all=bdgs_hisilicon, vol_all=[vol_intel] * len(bdgs_hisilicon), params=params))


def test_bp():
    params = {"type_pkg": "SI", "w_power": 0, "w_perf": 0, "w_cost": 1}

    bdgs_amd = dataset.AMD()
    vol_intel = 10 * 1000 * 1000
    print(
        baseline.balanced_partition(bdg_all=bdgs_amd,
                                    vol_all=[vol_intel] * len(bdgs_amd),
                                    tool="metis",
                                    dir_tool=dir_metis,
                                    params=params))


def test_gn():
    params = {}
    params["num_cpu"] = 32
    params["indi_pnum"] = 2
    params["type_pkg"] = "SI"
    params["w_power"] = 0
    params["w_perf"] = 0
    params["w_cost"] = 1

    # params of HoliSolver
    params["holi_pnum"] = 1
    params["holi_num_init_sample"] = 20
    params["holi_max_try"] = 10
    params["holi_T_start"] = 1
    params["holi_T_end"] = 0.5
    params["holi_alpha"] = 0.95

    cpu = Block(name="cpu", area=8, node=7)
    gpu = Block(name="gpu", area=12, node=7)
    cpu_num = [40, 38, 32, 28, 26, 24, 18, 16, 12, 8]
    gpu_num = [40, 8, 4, 4, 4, 5, 1, 2, 8, 8]
    bdg_all = []
    for n_cpu, n_gpu in zip(cpu_num, gpu_num):
        bdg = nx.DiGraph()
        bdg.add_nodes_from(range(n_cpu), block=cpu)
        bdg.add_nodes_from(range(n_cpu, n_cpu + n_gpu), block=gpu)
        bdg_all.append(bdg)
    vol = 5000 * 1000
    vol_all = [vol] * len(cpu_num)

    baseline.get_cn_paper(bdg_all=bdg_all, vol_all=vol_all, params=params)
    # baseline.get_cn_naive(bdg_all=bdg_all, vol_all=vol_all, params=params)
    bc_all = [[{cpu, gpu}] for _ in range(len(bdg_all))]
    cn_all = [[1] for _ in range(len(bdg_all))]
    hsolver = HoliSolver(bdg_all=bdg_all, vol_all=vol_all, bc_all=bc_all, cn_all=cn_all, dir_log=None, params=params)
    print(
        hsolver.adjust_cn(bc_target={gpu, cpu},
                          state_bc=hsolver.state_bc_cur,
                          state_cn=hsolver.state_cn_cur,
                          ppc=hsolver.ppc_cur))


def test_chopin():
    params = {
        "num_cpu": 32,
        "pnum": 2,
        "dir_log_root": osp.join(dir_root, "log/test"),
        "indi_max_try": 10,
        "type_pkg": "SI",
        "indi_T_start": 1,
        "indi_T_end": 0.01,
        "indi_alpha": 0.95,
        "w_power": 0,
        "w_perf": 0,
        "w_cost": 1
    }
    bdgs_amd = dataset.AMD()
    vol_amd = 500 * 1000
    baseline.chopin(bdg_all=bdgs_amd, vol_all=[vol_amd] * len(bdgs_amd), x=1.1, params=params)
    vol_amd = 20 * 1000 * 1000
    baseline.chopin(bdg_all=bdgs_amd, vol_all=[vol_amd] * len(bdgs_amd), x=1.1, params=params)


def test_reuse_first():
    params = {
        "type_pkg": "SI",
        "indi_T_start": 1,
        "indi_T_end": 0.01,
        "indi_alpha": 0.95,
        "w_power": 0,
        "w_perf": 0,
        "w_cost": 1
    }
    bdgs_hisilicon = dataset.HiSilicon()
    vol_hisilicon = 500 * 1000
    print(
        baseline.reuse_first(dir_cplex=dir_cplex,
                             bdg_all=bdgs_hisilicon,
                             vol_all=[vol_hisilicon] * len(bdgs_hisilicon),
                             params=params))


if __name__ == "__main__":
    test_bp()