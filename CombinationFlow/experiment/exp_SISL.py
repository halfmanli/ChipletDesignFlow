import os
import pickle
from framework import GIA
from framework.ChipletSys import ChipletSys
from multiprocessing import Pool


def joint_opt_SISL(dir_log,
                   csys,
                   placement,
                   topo_type,
                   weight,
                   tile_size,
                   T_th,
                   pid,
                   topo_graph=None,
                   rtp=None,
                   avg_lat_packet=None,
                   intp_type=None):
    assert tile_size == 0.001
    assert T_th == 85

    def cost_func(csys, placement, stage, nv_dict):
        therm = csys.eval_thermal(dir_hotspot=dir_hotspot, tile_size=tile_size, placement=placement, visualize=False)
        PPP = GIA.SISL(dir_dsent=dir_dsent,
                       dir_booksim=dir_booksim,
                       topo_type=topo_type,
                       csys=csys,
                       placement=placement,
                       tile_size=tile_size,
                       cfg_booksim={
                           "sim_cycle": 10000,
                           "num_vcs": 4,
                           "vc_buf_size": 4
                       },
                       topo_graph=topo_graph,
                       rtp=rtp,
                       avg_lat_packet=avg_lat_packet,
                       intp_type=intp_type)  # power, performance, package cost
        power_eu = PPP["power_eu"]
        perf = PPP["perf"]
        pcost = PPP["pcost"]
        mlayout = None
        if stage == "init_joint":
            cvi = (1, {"thermal": therm.max(), "thermal_map": therm, "pcost": pcost, "mlayout": mlayout, **PPP})
            return cvi
        elif stage == "joint_opt":
            cv = weight["power_eu"] * power_eu / nv_dict["power_eu"] + weight["perf"] * perf / nv_dict["perf"] + weight[
                "pcost"] * pcost / (nv_dict["pcost"] + 1e-20) + weight["thermal"] * max(therm.max() - T_th, 0)
            ci = {"thermal": therm.max(), "thermal_map": therm, "pcost": pcost, "mlayout": mlayout, **PPP}
            return (cv, ci)
        else:
            assert False

    placer = GIA.PlacerSA(csys=csys, cost_func=cost_func, cfg_algo={"init_n_j": 20, "T_end_j": 0.25}, pid=pid, dir_log=dir_log)
    placer.jopt(placement=placement)  # joint optimization
    return placer.best_sol


def run_SISL(topo_type, min_max_cpl, weight, T_th, intp_type=None):
    """
        run SISL experiment
        weight: weight factor dict of power, perf, cost, max(T-T_th, 0)
    """
    with open(os.path.join(dir_data, "dataset/sys_{}_{}_topt.pkl".format(*min_max_cpl)), "rb") as f:
        dt = pickle.load(f)

    num_cpu = 50
    pool = Pool(processes=num_cpu)
    res = []

    for idx_d, d in enumerate(dt):
        if idx_d >= 70:
            continue
        csys = ChipletSys(W=d["W_intp"],
                          H=d["H_intp"],
                          chiplets=d["chiplets"],
                          task_graph=d["task_graph"],
                          pin_map=d["pin_map"])
        placement = d["placement"][0]
        topo_graph = d["topo_graph"]
        rtp = d["rtp"]
        avg_lat_packet = d["avg_lat_packet"]
        if topo_type == "app":
            dir_log_root = os.path.join(dir_data, "SISL/{}_{}/log_{}_{}".format(topo_type, intp_type, *min_max_cpl))
        else:
            dir_log_root = os.path.join(dir_data, "SISL/{}/log_{}_{}".format(topo_type, *min_max_cpl))
        dir_log = os.path.join(dir_log_root, "{}".format(idx_d))
        os.mkdir(dir_log)
        for pid in range(2):
            res.append(
                pool.apply_async(joint_opt_SISL,
                                 args=(dir_log, csys, placement, topo_type, weight, d["tile_size"], T_th, pid, topo_graph, rtp,
                                       avg_lat_packet, intp_type)))
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

    if True:
        # run_SISL(topo_type="app",
        #          min_max_cpl=(0, 0),
        #          weight={
        #              "power_eu": 0.5,
        #              "perf": 0.5,
        #              "thermal": 0.1,
        #              "pcost": 0
        #          },
        #          T_th=85,
        #          intp_type="active")
        # print("app active done")
        # run_SISL(topo_type="app",
        #          min_max_cpl=(0, 0),
        #          weight={
        #              "power_eu": 0.5,
        #              "perf": 0.5,
        #              "thermal": 0.1,
        #              "pcost": 0
        #          },
        #          T_th=85,
        #          intp_type="passive")
        # print("app passive done")
        run_SISL(topo_type="bd",
                 min_max_cpl=(0, 0),
                 weight={
                     "power_eu": 0.5,
                     "perf": 0.5,
                     "thermal": 0.1,
                     "pcost": 0
                 },
                 T_th=85)
        print("bd done")
        run_SISL(topo_type="ft",
                 min_max_cpl=(0, 0),
                 weight={
                     "power_eu": 0.5,
                     "perf": 0.5,
                     "thermal": 0.1,
                     "pcost": 0
                 },
                 T_th=85)
        print("ft done")
        run_SISL(topo_type="mesh",
                 min_max_cpl=(0, 0),
                 weight={
                     "power_eu": 0.5,
                     "perf": 0.5,
                     "thermal": 0.1,
                     "pcost": 0
                 },
                 T_th=85)
        print("mesh done")