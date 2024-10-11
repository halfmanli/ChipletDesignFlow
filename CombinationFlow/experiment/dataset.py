from multiprocessing import Pool
import os
import random
import shutil
import tempfile
from framework import NoC, Sunfloor, GIA
from framework.ChipletSys import ChipletSys, Chiplet
import numpy as np
import itertools as itl
# import cplex
import pickle
import math
from rectpack import newPacker

cpl_lib = [
    {
        "code": "CPU_0",
        "real_name": "Intel Atom N270",
        "w": 7.94 * 1e-3,  # unit: meter
        "h": 3.27 * 1e-3,
        "power": 2.5,  # Watt
        "bw": 1.57 * 8  # Gbit/s
    },
    {
        "code": "CPU_1",
        "real_name": "AMD Ryzen5 3600",
        "w": 8.6 * 1e-3,
        "h": 8.6 * 1e-3,
        "power": 65,
        "bw": 47.65 * 8
    },
    {
        "code": "CPU_2",
        "real_name": "waferscale processor system",
        "w": 3.15 * 1e-3,
        "h": 2.4 * 1e-3,
        "power": 0.175,
        "bw": 9.6 * 8  # 9.83 * 1000 / 1024
    },
    {
        "code": "DSP",
        "real_name": "dsp",
        "w": 2.5 * 1e-3,
        "h": 2.5 * 1e-3,
        "power": 0.5,
        "bw": 61.6 * 8
    },
    {
        "code": "DL_ACC_0",
        "real_name": "Simba",
        "w": 2.5 * 1e-3,
        "h": 2.4 * 1e-3,
        "power": 3.6,
        "bw": 100 * 8
    },
    {
        "code": "DL_ACC_1",
        "real_name": "Google TPU v1",
        "w": 18.2 * 1e-3,
        "h": 18.2 * 1e-3,
        "power": 75,
        "bw": 34 * 8
    },
    {
        "code": "DL_ACC_2",
        "real_name": "Google TPU v2",
        "w": 24.7 * 1e-3,
        "h": 24.7 * 1e-3,
        "power": 280,
        "bw": 700 * 8
    },
    {
        "code": "GPU_0",
        "real_name": "Nvidia GeForce GT 1010",
        "w": 8.6 * 1e-3,
        "h": 8.6 * 1e-3,
        "power": 30,
        "bw": 48.06 * 8
    },
    {
        "code": "GPU_1",
        "real_name": "AMD Navi 24",
        "w": 10.3 * 1e-3,
        "h": 10.3 * 1e-3,
        "power": 107,
        "bw": 144 * 8
    },
    {
        "code": "GPU_2",
        "real_name": "Nvidia GeForce GTX 1080",
        "w": 17.7 * 1e-3,
        "h": 17.7 * 1e-3,
        "power": 180,
        "bw": 320.3 * 8
    },
    {
        "code": "SRAM",
        "real_name": "waferscale processor system",
        "w": 3.15 * 1e-3,
        "h": 1.1 * 1e-3,
        "power": 0.175,
        "bw": 9.6 * 8
    },
    {
        "code": "DRAM",
        "real_name": "ddr4 3200",
        "w": 8.75 * 1e-3,
        "h": 8.75 * 1e-3,
        "power": 20,
        "bw": 25.6 * 8
    }
]


def gen_sys_dt(dir_tgff, dir_cplex, dir_dt, sys_cnt, cpl_lib, bw, cfg_tgff, tile_size, cfg_sys=None, clean=True):
    """
        generate the dataset of chiplets
        sys_cnt: the number of generated chiplet system
        cpl_lib: raw chiplets, list of dict
        bw: bandwidth of link, Gbit/s
        cfg_tgff: "tg_cnt" decides the retry times of one iteration of specific chipelts
        cfg_sys: configuration dict
    """
    assert bw == 128
    # configurations of chiplets
    W_intp = cfg_sys["W_intp"]  # width of interposer, unit: tile
    H_intp = cfg_sys["H_intp"]  # height of interposer, unit: tile
    max_area_sys = cfg_sys["max_area_sys"]  # max of total area of chiplets, unit is tile^2
    max_num_cpl = cfg_sys["max_num_cpl"]  # max number of chiplets in system
    min_num_cpl = cfg_sys["min_num_cpl"]  # min num of chiplets in system
    max_pd_sys = cfg_sys["max_pd_sys"]  # Watt/tile^2, max power density of whole system
    min_pd_sys = cfg_sys["min_pd_sys"]  # Watt/tile^2, min power density of whole system

    min_power_sys = int(min_pd_sys * W_intp * H_intp)
    max_power_sys = int(max_pd_sys * W_intp * H_intp)

    # preprocess of cpl_lib
    cpl_cdd = []
    for cpl in cpl_lib:
        cpl_zoom = {}
        cpl_zoom["code"] = cpl["code"]
        cpl_zoom["real_name"] = cpl["real_name"]
        cpl_zoom["w"] = math.ceil(cpl["w"] / tile_size)  # unit is tile
        cpl_zoom["h"] = math.ceil(cpl["h"] / tile_size)
        ratio_zoom = (cpl_zoom["w"] * tile_size * cpl_zoom["h"] * tile_size) / (cpl["w"] * cpl["h"])
        cpl_zoom["power"] = cpl["power"] * ratio_zoom
        pin_pos_avail = list(
            itl.chain(itl.product(range(cpl_zoom["w"] - 1), [cpl_zoom["h"] - 1]),
                      itl.product([cpl_zoom["w"] - 1], range(cpl_zoom["h"] - 1, 0, -1)),
                      itl.product(range(cpl_zoom["w"] - 1, 0, -1), [0]), itl.product([0], range(cpl_zoom["h"] - 1))))
        num_pin = math.ceil(cpl["bw"] / (bw * 2))  # bandwidth of chiplet = read + write
        assert num_pin <= len(pin_pos_avail)
        idx_pin = [round(idx_p * (len(pin_pos_avail) / num_pin)) for idx_p in range(num_pin)]
        assert len(idx_pin) == len(set(idx_pin))
        cpl_zoom["pins"] = [pin_pos_avail[idx] for idx in idx_pin]
        cpl_cdd.append(cpl_zoom)

    dt = []  # dataset
    scnt = 0
    while scnt < sys_cnt:
        flg_fail = True
        while True:
            max_single_cpl = 5  # max number of chiplets for single type
            dir_tmp = tempfile.mkdtemp(dir=os.path.join(dir_cplex, "build"))
            path_lp = os.path.join(dir_tmp, "solve.lp")
            with open(path_lp, "w") as f:
                f.write("Minimize z\n")
                f.write("Subject To\n")
                weight = []
                for _ in range(len(cpl_cdd)):
                    for __ in range(max_single_cpl):
                        weight.append(random.randint(-1e5, 1e5))
                f.write("z " + " ".join([
                    "{:+} n_{}_{}".format(weight[idx_cpl * max_single_cpl + j], idx_cpl, j) for idx_cpl in range(len(cpl_cdd))
                    for j in range(max_single_cpl)
                ]) + " >= 0\n")
                f.write("z " + " ".join([
                    "{:+} n_{}_{}".format(-weight[idx_cpl * max_single_cpl + j], idx_cpl, j) for idx_cpl in range(len(cpl_cdd))
                    for j in range(max_single_cpl)
                ]) + " >= 0\n")

                f.write(" + ".join([
                    "{} n_{}_{}".format(cpl_cdd[idx_cpl]["w"] * cpl_cdd[idx_cpl]["h"], idx_cpl, j)
                    for idx_cpl in range(len(cpl_cdd)) for j in range(max_single_cpl)
                ]) + " <= {}".format(max_area_sys) + "\n")
                f.write(" + ".join([
                    "{} n_{}_{}".format(cpl_cdd[idx_cpl]["power"], idx_cpl, j) for idx_cpl in range(len(cpl_cdd))
                    for j in range(max_single_cpl)
                ]) + " <= {}".format(max_power_sys) + "\n")
                f.write(" + ".join([
                    "{} n_{}_{}".format(cpl_cdd[idx_cpl]["power"], idx_cpl, j) for idx_cpl in range(len(cpl_cdd))
                    for j in range(max_single_cpl)
                ]) + " >= {}".format(min_power_sys) + "\n")
                f.write(
                    " + ".join(["n_{}_{}".format(idx_cpl, j) for idx_cpl in range(len(cpl_cdd))
                                for j in range(max_single_cpl)]) + " >= {}".format(min_num_cpl) + "\n")
                f.write(
                    " + ".join(["n_{}_{}".format(idx_cpl, j) for idx_cpl in range(len(cpl_cdd))
                                for j in range(max_single_cpl)]) + " <= {}".format(max_num_cpl) + "\n")
                f.write("Binary\n")
                f.write(
                    " ".join(["n_{}_{}".format(idx_cpl, j) for idx_cpl in range(len(cpl_cdd))
                              for j in range(max_single_cpl)]) + "\n")
                f.write("End")

            cpx = cplex.Cplex(path_lp)
            cpx.set_log_stream(None)
            cpx.set_error_stream(None)
            cpx.set_warning_stream(None)
            cpx.set_results_stream(None)
            cpx.solve()
            cpls_sel = []
            for idx_cpl in range(len(cpl_cdd)):
                for j in range(max_single_cpl):
                    if cpx.solution.get_values("n_{}_{}".format(idx_cpl, j)):
                        cpls_sel.append(cpl_cdd[idx_cpl])
            if clean:
                shutil.rmtree(dir_tmp)
            num_cpl = len(cpls_sel)
            break

        cfg_tgff["seed"] = random.randint(0, 1e5)
        gs = NoC.gen_task_tgff(dir_tgff=dir_tgff, bw=bw, cfg=cfg_tgff, filter=None, clean=clean)
        for g in gs:  # task graph
            pgs = []  # pins in same group are unconnected: [[pin 0, pin 1, ...], ..., [pin x, pin y, ...]]
            for p in g:
                flg_connected = False
                if p == 0:
                    pg_this = [0]
                else:
                    for p_t in pg_this:
                        if g.has_edge(p, p_t) or g.has_edge(p_t, p):
                            pgs.append(pg_this)
                            pg_this = [p]
                            flg_connected = True
                            break
                    if not flg_connected:
                        pg_this.append(p)
            pgs.append(pg_this)

            while True:
                flg_fail = True  # failed to increase/decrease node group
                num_pg = len(pgs)
                if num_pg > num_cpl:  # merge two unconnected groups
                    i_j_pg = [(i, j) for i in range(num_pg) for j in range(i + 1, num_pg)]
                    i_j_pg = sorted(i_j_pg, key=lambda i_j: len(pgs[i_j[0]]) + len(pgs[i_j[1]]))  # merge the smallest
                    for i_pg, j_pg in i_j_pg:
                        flg_cnted = False  # connected flag
                        for i_c, j_c in itl.product(pgs[i_pg], pgs[j_pg]):
                            if g.has_edge(i_c, j_c) or g.has_edge(j_c, i_c):
                                flg_cnted = True
                                break
                        if not flg_cnted:
                            pgs_ = [pgs[idx_pg] for idx_pg in range(num_pg) if idx_pg != i_pg and idx_pg != j_pg]
                            pgs = pgs_ + [pgs[i_pg] + pgs[j_pg]]
                            flg_fail = False
                            break
                    if flg_fail:
                        break

                elif num_pg < num_cpl:
                    idx_pgs = [i for i in range(num_pg) if len(pgs[i]) > 1]
                    if not idx_pgs:
                        break
                    idx_pg = np.random.choice(idx_pgs)  # the node group to be split
                    pg_split = pgs[idx_pg]
                    np.random.shuffle(pg_split)
                    pgs = pgs[:idx_pg] + pgs[idx_pg + 1:] + [pg_split[0::2], pg_split[1::2]]
                else:
                    flg_fail = False
                    break

            if flg_fail:
                continue
            else:
                assert sum([len(pg) for pg in pgs]) == len(g)
                assert set([p for pg in pgs for p in pg]) == set(range(len(g)))
                assert len(cpls_sel) == len(pgs)

            # map cores in task graph to pins at peripheral position
            cpls_sel = sorted(cpls_sel, key=lambda cpl: len(cpl["pins"]))
            pgs = sorted(pgs, key=lambda pg: len(pg))

            flg_fail = False
            for idx in range(num_cpl):
                if len(cpls_sel[idx]["pins"]) < len(pgs[idx]):  # too many NIs
                    flg_fail = True
                    break

            if not flg_fail:
                # generate chiplets
                chiplets = [
                    Chiplet(name=cpls_sel[idx_cpl]["code"] + "__{}".format(idx_cpl),
                            w=cpls_sel[idx_cpl]["w"],
                            h=cpls_sel[idx_cpl]["h"],
                            power=cpls_sel[idx_cpl]["power"],
                            pins=cpls_sel[idx_cpl]["pins"]) for idx_cpl in range(num_cpl)
                ]

                packer = newPacker()
                for idx_cpl, cpl in enumerate(chiplets):
                    packer.add_rect(width=cpl.w_orig, height=cpl.h_orig, rid=idx_cpl)
                packer.add_bin(width=W_intp, height=H_intp, count=1)
                packer.pack()
                all_rects = packer.rect_list()
                if not len(all_rects) == len(chiplets):
                    break

                pin_map = {}  # mapping from task graph to (idx_cpl, idx_pin_cpl)
                for idx_cpl in range(num_cpl):  # add nodes/pins
                    shuffle_pins = sorted(list(range(len(cpls_sel[idx_cpl]["pins"]))), key=lambda e: random.random())
                    for idx_pg_cpl in range(len(pgs[idx_cpl])):
                        pin_map[pgs[idx_cpl][idx_pg_cpl]] = (idx_cpl, shuffle_pins[idx_pg_cpl])
                for (u, v, _) in g.edges(data=True):
                    assert pin_map[u][0] != pin_map[v][0]  # pins having edges must not in same chiplet
                dt.append({
                    "W_intp": W_intp,
                    "H_intp": H_intp,
                    "chiplets": chiplets,
                    "tile_size": tile_size,
                    "task_graph": g,
                    "pin_map": pin_map
                })
                scnt += 1
                print("success")
                break

    with open(os.path.join(dir_dt, "sys_{}_{}.pkl".format(min_num_cpl, max_num_cpl)), "wb") as outp:
        pickle.dump(dt, outp, pickle.HIGHEST_PROTOCOL)


def single_opt(csys, tile_size, dir_log, pid):

    def cost_func(csys, placement, stage, nv_dict):
        therm = csys.eval_thermal(dir_hotspot=dir_hotspot, tile_size=tile_size, placement=placement, visualize=False)
        if stage == "init_therm":
            return (therm.max(), {"thermal": therm.max(), "thermal_map": therm})
        elif stage == "therm_opt":
            return (therm.max() / nv_dict["thermal"], {"thermal": therm.max(), "thermal_map": therm})
        else:
            assert False

    placer = GIA.PlacerSA(csys=csys, cost_func=cost_func, cfg_algo={}, pid=pid, dir_log=dir_log)
    placer.topt()  # thermal optimization
    with open(os.path.join(dir_log, "{}.pkl".format(pid)), "wb") as out_p:
        pickle.dump(placer.best_sol[0], out_p)
    return placer.best_sol


def topt(tile_size, min_num_cpl, max_num_cpl):
    assert tile_size == 1e-3
    dir_log_root = os.path.join(dir_root, "data/topt/log_{}_{}".format(min_num_cpl, max_num_cpl))
    path_dt = os.path.join(dir_root, "data/dataset/sys_{}_{}.pkl".format(min_num_cpl, max_num_cpl))
    csystems = []
    with open(path_dt, "rb") as f:
        dt = pickle.load(f)  # data point
        for d in dt:
            csystems.append(
                ChipletSys(W=d["W_intp"],
                           H=d["H_intp"],
                           chiplets=d["chiplets"],
                           task_graph=d["task_graph"],
                           pin_map=d["pin_map"]))
    num_cpu = 40
    pnum = 5  # process number
    pool = Pool(processes=num_cpu)
    res = []
    for idx_csys, csys in enumerate(csystems):
        dir_log = os.path.join(dir_log_root, "{}".format(idx_csys))
        os.mkdir(dir_log)
        for pid in range(pnum):
            res.append(pool.apply_async(single_opt, args=(csys, tile_size, dir_log, pid)))
    pool.close()
    pool.join()
    for r in res:
        print(r.get())


def after_topt_single(min_num_cpl, max_num_cpl, idx_d, d):
    csys = ChipletSys(W=d["W_intp"], H=d["H_intp"], chiplets=d["chiplets"], task_graph=d["task_graph"], pin_map=d["pin_map"])
    d["topo_graph"], d["rtp"] = Sunfloor.sunfloor(dir_chaco=dir_chaco,
                                                  dir_dsent=dir_dsent,
                                                  task_graph=d["task_graph"],
                                                  max_port=4,
                                                  bw=128)
    PPA_booksim = NoC.eval_PPA_booksim(dir_booksim=dir_booksim,
                                       task_graph=csys.task_graph,
                                       topo_graph=d["topo_graph"],
                                       rtp=d["rtp"],
                                       cfg={
                                           "sim_cycle": 10000,
                                           "num_vcs": 4,
                                           "vc_buf_size": 4
                                       },
                                       clean=True)
    d["avg_lat_packet"] = PPA_booksim["avg_lat_packet"]
    d["placement"] = []
    for i in range(5):
        path_placement = os.path.join(dir_root, "data/topt/log_{}_{}/{}/{}.pkl".format(min_num_cpl, max_num_cpl, idx_d, i))
        if not os.path.exists(path_placement):
            continue
        with open(path_placement, "rb") as f:
            placement = pickle.load(f)
        therm = csys.eval_thermal(dir_hotspot, tile_size=d["tile_size"], placement=placement, visualize=False, cfg=None)
        if therm.max() <= 88:  # TODO:
            d["placement"].append(placement)

    print("{} done".format(idx_d))
    if d["placement"]:  # lower than 85
        return d
    else:
        return None


def after_topt(min_num_cpl, max_num_cpl):
    """
        generate datasets after thermal optimization
    """
    path_dt = os.path.join(dir_data, "dataset/sys_{}_{}.pkl".format(min_num_cpl, max_num_cpl))
    with open(path_dt, "rb") as f:
        num_cpu = 40
        pool = Pool(processes=num_cpu)
        res = []
        dt = pickle.load(f)  # data point
        for idx_d, d in enumerate(dt):
            res.append(pool.apply_async(after_topt_single, args=(min_num_cpl, max_num_cpl, idx_d, d)))
        pool.close()
        pool.join()
        dt_topt = []
        for r in res:
            d = r.get()
            if d is not None:
                dt_topt.append(d)
    path_dt_topt = os.path.join(dir_data, "dataset/sys_{}_{}_topt.pkl".format(min_num_cpl, max_num_cpl))
    with open(path_dt_topt, "wb") as f:
        pickle.dump(dt_topt, f)


if __name__ == "__main__":
    dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    dir_tgff = os.path.join(dir_root, "tool/tgff")
    dir_cplex = os.path.join(dir_root, "tool/cplex")
    dir_dsent = os.path.join(dir_root, "tool/dsent")
    dir_chaco = os.path.join(dir_root, "tool/chaco")
    dir_booksim = os.path.join(dir_root, "tool/booksim")
    dir_data = os.path.join(dir_root, "data")
    dir_hotspot = os.path.join(dir_root, "tool/hotspot")

    if False:
        gen_sys_dt(dir_tgff=dir_tgff,
                   dir_cplex=dir_cplex,
                   dir_dt=os.path.join(dir_data, "dataset"),
                   sys_cnt=100,
                   cpl_lib=cpl_lib,
                   bw=128,
                   cfg_tgff={
                       "task_cnt": (25, 5),
                       "tg_cnt": 200
                   },
                   tile_size=1e-3,
                   cfg_sys={
                       "W_intp": 20,
                       "H_intp": 20,
                       "max_area_sys": 20 * 20,
                       "max_num_cpl": 10,
                       "min_num_cpl": 6,
                       "max_pd_sys": 0.7,
                       "min_pd_sys": 0.35
                   },
                   clean=True)

        gen_sys_dt(dir_tgff=dir_tgff,
                   dir_cplex=dir_cplex,
                   dir_dt=os.path.join(dir_data, "dataset"),
                   sys_cnt=100,
                   cpl_lib=cpl_lib,
                   bw=128,
                   cfg_tgff={
                       "task_cnt": (35, 5),
                       "tg_cnt": 200
                   },
                   tile_size=1e-3,
                   cfg_sys={
                       "W_intp": 30,
                       "H_intp": 30,
                       "max_area_sys": 0.8 * 30 * 30,
                       "max_num_cpl": 15,
                       "min_num_cpl": 11,
                       "max_pd_sys": 0.65,
                       "min_pd_sys": 0.35
                   },
                   clean=True)

        gen_sys_dt(dir_tgff=dir_tgff,
                   dir_cplex=dir_cplex,
                   dir_dt=os.path.join(dir_data, "dataset"),
                   sys_cnt=100,
                   cpl_lib=cpl_lib,
                   bw=128,
                   cfg_tgff={
                       "task_cnt": (45, 5),
                       "tg_cnt": 200
                   },
                   tile_size=1e-3,
                   cfg_sys={
                       "W_intp": 40,
                       "H_intp": 40,
                       "max_area_sys": 0.7 * 40 * 40,
                       "max_num_cpl": 20,
                       "min_num_cpl": 16,
                       "max_pd_sys": 0.6,
                       "min_pd_sys": 0.35
                   },
                   clean=True)

    if True:
        # topt(tile_size=1e-3, min_num_cpl=6, max_num_cpl=10)
        # topt(tile_size=1e-3, min_num_cpl=11, max_num_cpl=15)
        # topt(tile_size=1e-3, min_num_cpl=16, max_num_cpl=20)
        topt(tile_size=1e-3, min_num_cpl=0, max_num_cpl=0)

    if True:
        # after_topt(min_num_cpl=6, max_num_cpl=10)
        # after_topt(min_num_cpl=11, max_num_cpl=15)
        # after_topt(min_num_cpl=16, max_num_cpl=20)
        after_topt(min_num_cpl=0, max_num_cpl=0)
