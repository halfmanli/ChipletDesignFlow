import os
import pickle
from framework import GIA
from framework.ChipletSys import ChipletSys
from multiprocessing import Pool


def joint_opt_ActiveGIA(dir_log, csys, placement, topo_graph, rtp, avg_lat_packet, weight, tile_size, T_th, pid):
    assert tile_size == 0.001
    assert T_th == 85

    def cost_func(csys, placement, stage, nv_dict):
        therm = csys.eval_thermal(dir_hotspot=dir_hotspot, tile_size=tile_size, placement=placement, visualize=False)
        PPP = GIA.ActiveGIA(dir_dsent=dir_dsent,
                            csys=csys,
                            placement=placement,
                            topo_graph=topo_graph,
                            rtp=rtp,
                            tile_size=tile_size,
                            bw=128,
                            freq=1e9)
        if PPP is None:  # failed to generate mlayout
            cvi = (1e9, {
                "thermal": therm.max(),
                "thermal_map": therm,
                "power": 1e9,
                "power_eu": 1e9,
                "perf": 1e9,
                "pcost": 1e9,
                "mlayout": None
            })
            return cvi
        else:
            power_eu = PPP["power_eu"]  # consider esd and ubump
            perf = avg_lat_packet + PPP["lat_penal"]
            pcost = PPP["pcost"]
            if stage == "init_joint":
                cvi = (1, {"thermal": therm.max(), "thermal_map": therm, "perf": perf, **PPP})
                return cvi
            elif stage == "joint_opt":
                cv = weight["power_eu"] * power_eu / nv_dict["power_eu"] + weight["perf"] * perf / nv_dict["perf"] + weight[
                    "pcost"] * pcost / (nv_dict["pcost"] + 1e-20) + weight["thermal"] * max(therm.max() - T_th, 0)
                ci = {"thermal": therm.max(), "thermal_map": therm, "perf": perf, **PPP}
                return (cv, ci)
            else:
                assert False

    placer = GIA.PlacerSA(csys=csys, cost_func=cost_func, cfg_algo={"init_n_j": 20, "T_end_j": 0.25}, pid=pid, dir_log=dir_log)
    placer.jopt(placement=placement)
    return placer.best_sol


def run_ActiveGIA(min_max_cpl, weight, T_th):
    """
        run SISL experiment
        weight: weight factor dict of power, perf, cost, max(T-T_th, 0)
    """
    with open(os.path.join(dir_data, "dataset/sys_{}_{}_topt.pkl".format(*min_max_cpl)), "rb") as f:
        dt = pickle.load(f)

    num_cpu = 32  # process number
    pool = Pool(processes=num_cpu)
    res = []

    for idx_d, d in enumerate(dt):
        csys = ChipletSys(W=d["W_intp"],
                          H=d["H_intp"],
                          chiplets=d["chiplets"],
                          task_graph=d["task_graph"],
                          pin_map=d["pin_map"])
        placement = d["placement"][0]
        dir_log_root = os.path.join(dir_data, "ActiveGIA/log_{}_{}".format(*min_max_cpl))
        dir_log = os.path.join(dir_log_root, "{}".format(idx_d))
        os.mkdir(dir_log)
        for pid in range(2):
            res.append(
                pool.apply_async(joint_opt_ActiveGIA,
                                 args=(dir_log, csys, placement, d["topo_graph"], d["rtp"], d["avg_lat_packet"], weight,
                                       d["tile_size"], T_th, pid)))
    pool.close()
    pool.join()
    for r in res:
        r.get()


if __name__ == "__main__":
    dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    dir_dsent = os.path.join(dir_root, "tool/dsent")
    dir_chaco = os.path.join(dir_root, "tool/chaco")
    dir_booksim = os.path.join(dir_root, "tool/booksim")
    dir_hotspot = os.path.join(dir_root, "tool/hotspot")
    dir_data = os.path.join(dir_root, "data")

    # run_ActiveGIA(min_max_cpl=(6, 10), weight={"power_eu": 0.5, "perf": 0.5, "thermal": 0.1, "pcost": 0}, T_th=85)
    # run_ActiveGIA(min_max_cpl=(11, 15), weight={"power_eu": 0.5, "perf": 0.5, "thermal": 0.1, "pcost": 0}, T_th=85)
    # run_ActiveGIA(min_max_cpl=(16, 20), weight={"power_eu": 0.5, "perf": 0.5, "thermal": 0.1, "pcost": 0}, T_th=85)
    run_ActiveGIA(min_max_cpl=(0, 0), weight={"power_eu": 0.5, "perf": 0.5, "thermal": 0.1, "pcost": 0}, T_th=85)