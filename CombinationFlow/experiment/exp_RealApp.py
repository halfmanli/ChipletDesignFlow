from collections import defaultdict
import math
from bidict import bidict
import numpy as np
import os
import pickle
import networkx as nx
from framework import NoC, Sunfloor
from framework.ChipletSys import Chiplet
from framework.GIA import CType


def add_link(arr, arr_name, src, dst, value):
    arr[arr_name.index(src)][arr_name.index(dst)] = value


def app1():
    MMS_name = [
        "ASIC1", "ASIC2", "ASIC3", "ASIC4", "DSP1", "DSP2", "DSP3", "DSP4", "DSP5", "DSP6", "DSP7", "DSP8", "CPU", "MEM1",
        "MEM2", "MEM3"
    ]
    MMS_mapping = {
        "ASIC1": cpu,
        "ASIC2": cpu,
        "ASIC3": cpu,
        "ASIC4": cpu,
        "DSP1": dsp,
        "DSP2": dsp,
        "DSP3": dsp,
        "DSP4": dsp,
        "DSP5": dsp,
        "DSP6": dsp,
        "DSP7": dsp,
        "DSP8": dsp,
        "CPU": cpu,
        "MEM1": dram,
        "MEM2": dram,
        "MEM3": dram
    }
    MMS = np.zeros((16, 16))
    add_link(MMS, MMS_name, "ASIC1", "ASIC2", 25)
    add_link(MMS, MMS_name, "ASIC1", "DSP8", 25)
    add_link(MMS, MMS_name, "ASIC2", "ASIC3", 764)
    add_link(MMS, MMS_name, "ASIC2", "MEM2", 640)
    add_link(MMS, MMS_name, "ASIC2", "ASIC1", 80)
    add_link(MMS, MMS_name, "ASIC3", "DSP8", 641)
    add_link(MMS, MMS_name, "ASIC3", "DSP4", 144)
    add_link(MMS, MMS_name, "ASIC4", "DSP1", 33848)
    add_link(MMS, MMS_name, "ASIC4", "CPU", 197)
    add_link(MMS, MMS_name, "CPU", "MEM1", 38016)
    add_link(MMS, MMS_name, "CPU", "MEM3", 38016)
    add_link(MMS, MMS_name, "CPU", "ASIC3", 38016)
    add_link(MMS, MMS_name, "DSP1", "DSP2", 33848)
    add_link(MMS, MMS_name, "DSP1", "CPU", 20363)
    add_link(MMS, MMS_name, "DSP2", "ASIC2", 33848)
    add_link(MMS, MMS_name, "DSP2", "DSP1", 20363)
    add_link(MMS, MMS_name, "DSP3", "ASIC4", 38016)
    add_link(MMS, MMS_name, "DSP3", "DSP6", 7061)
    add_link(MMS, MMS_name, "DSP3", "DSP5", 7061)
    add_link(MMS, MMS_name, "DSP4", "DSP1", 3672)
    add_link(MMS, MMS_name, "DSP4", "CPU", 197)
    add_link(MMS, MMS_name, "DSP5", "DSP6", 26924)
    add_link(MMS, MMS_name, "DSP6", "ASIC2", 28248)
    add_link(MMS, MMS_name, "DSP7", "MEM2", 7065)
    add_link(MMS, MMS_name, "DSP8", "DSP7", 28265)
    add_link(MMS, MMS_name, "DSP8", "ASIC1", 80)
    add_link(MMS, MMS_name, "MEM1", "ASIC4", 116873)
    add_link(MMS, MMS_name, "MEM1", "CPU", 75205)
    add_link(MMS, MMS_name, "MEM2", "ASIC3", 7705)
    add_link(MMS, MMS_name, "MEM3", "CPU", 75584)
    MMS = np.ceil(MMS * 10 / 1000 / 1000 * 50).astype(int)  # MMS: unit is 10K
    task_graph = nx.DiGraph()
    task_graph.add_nodes_from(list(range(len(MMS))))  # This is important for chaco, TODO: fix this bug
    task_graph.add_edges_from([(i, j, {"comm": MMS[i][j]}) for i in range(len(MMS)) for j in range(len(MMS)) if MMS[i][j]])

    topo_graph, rtp = Sunfloor.sunfloor(dir_chaco=dir_chaco, dir_dsent=dir_dsent, task_graph=task_graph, max_port=4, bw=128)
    chiplets = []
    pin_map = dict()
    for i in range(len(task_graph)):
        ctype = MMS_mapping[MMS_name[i]]
        chiplets.append(Chiplet(name=ctype.name + "__{}".format(i), w=ctype.w, h=ctype.h, power=ctype.power, pins=ctype.pins))
        pin_map[i] = (i, 0)
    data = {
        "W_intp": 21,
        "H_intp": 21,
        "chiplets": chiplets,
        "tile_size": tile_size,
        "task_graph": task_graph,
        "pin_map": pin_map
    }
    return data


def app2():
    cholesky_name = [
        "caller", "potrf#1", "trsm#2", "trsm#1", "trsm#3", "syrk#2", "gemm#1", "gemm#3", "syrk#1", "gemm#2", "syrk#4",
        "potrf#2", "trsm#4", "trsm#5", "syrk#3", "gemm#4", "syrk#5", "potrf#3", "trsm#6", "syrk#6", "potrf#4"
    ]
    cholesky_mapping = {
        "caller": dram,
        "potrf#1": cpu,
        "trsm#2": cpu,
        "trsm#1": cpu,
        "trsm#3": cpu,
        "syrk#2": cpu,
        "gemm#1": cpu,
        "gemm#3": cpu,
        "syrk#1": cpu,
        "gemm#2": cpu,
        "syrk#4": cpu,
        "potrf#2": cpu,
        "trsm#4": cpu,
        "trsm#5": cpu,
        "syrk#3": cpu,
        "gemm#4": cpu,
        "syrk#5": cpu,
        "potrf#3": cpu,
        "trsm#6": cpu,
        "syrk#6": cpu,
        "potrf#4": cpu
    }
    n = len(cholesky_name)
    cholesky = np.zeros((n, n))
    add_link(cholesky, cholesky_name, "caller", "syrk#2", 20873440)
    add_link(cholesky, cholesky_name, "caller", "gemm#1", 41038912)
    add_link(cholesky, cholesky_name, "caller", "trsm#2", 41037367)
    add_link(cholesky, cholesky_name, "caller", "gemm#3", 41038912)
    add_link(cholesky, cholesky_name, "caller", "trsm#1", 41037367)
    add_link(cholesky, cholesky_name, "caller", "potrf#1", 20885368)
    add_link(cholesky, cholesky_name, "caller", "syrk#1", 20873440)
    add_link(cholesky, cholesky_name, "caller", "trsm#3", 41037367)
    add_link(cholesky, cholesky_name, "caller", "gemm#2", 41038912)
    add_link(cholesky, cholesky_name, "caller", "syrk#4", 20873440)
    add_link(cholesky, cholesky_name, "potrf#1", "trsm#2", 208044416)
    add_link(cholesky, cholesky_name, "potrf#1", "trsm#1", 208244421)
    add_link(cholesky, cholesky_name, "potrf#1", "trsm#3", 208044416)
    add_link(cholesky, cholesky_name, "trsm#2", "syrk#2", 4096411648)
    add_link(cholesky, cholesky_name, "trsm#2", "gemm#1", 40964096)
    add_link(cholesky, cholesky_name, "trsm#2", "gemm#3", 4096262144)
    add_link(cholesky, cholesky_name, "trsm#1", "gemm#1", 4096262144)
    add_link(cholesky, cholesky_name, "trsm#1", "syrk#1", 4096411648)
    add_link(cholesky, cholesky_name, "trsm#1", "gemm#2", 4096262144)
    add_link(cholesky, cholesky_name, "trsm#3", "gemm#3", 40964096)
    add_link(cholesky, cholesky_name, "trsm#3", "gemm#2", 40964096)
    add_link(cholesky, cholesky_name, "trsm#3", "syrk#4", 4096411648)
    add_link(cholesky, cholesky_name, "syrk#2", "syrk#3", 20822085)
    add_link(cholesky, cholesky_name, "gemm#1", "trsm#4", 40984101)
    add_link(cholesky, cholesky_name, "gemm#3", "gemm#4", 40984101)
    add_link(cholesky, cholesky_name, "syrk#1", "potrf#2", 20822089)
    add_link(cholesky, cholesky_name, "gemm#2", "trsm#5", 40984101)
    add_link(cholesky, cholesky_name, "syrk#4", "syrk#5", 20822085)
    add_link(cholesky, cholesky_name, "potrf#2", "trsm#4", 208044416)
    add_link(cholesky, cholesky_name, "potrf#2", "trsm#5", 208044416)
    add_link(cholesky, cholesky_name, "trsm#4", "syrk#3", 4096411648)
    add_link(cholesky, cholesky_name, "trsm#4", "gemm#4", 4096262144)
    add_link(cholesky, cholesky_name, "trsm#5", "gemm#4", 40964096)
    add_link(cholesky, cholesky_name, "trsm#5", "syrk#5", 4096411648)
    add_link(cholesky, cholesky_name, "syrk#3", "potrf#3", 20822089)
    add_link(cholesky, cholesky_name, "gemm#4", "trsm#6", 40984101)
    add_link(cholesky, cholesky_name, "syrk#5", "syrk#6", 20822085)
    add_link(cholesky, cholesky_name, "potrf#3", "trsm#6", 208044416)
    add_link(cholesky, cholesky_name, "trsm#6", "syrk#6", 4096411648)
    add_link(cholesky, cholesky_name, "syrk#6", "potrf#4", 20822089)

    cholesky = np.ceil(cholesky / (max(max(cholesky.sum(axis=0)), max(cholesky.sum(axis=1))) / 90)).astype(int)
    cholesky_max = cholesky.max()
    cholesky_min = cholesky.min()
    scale_max = cholesky_max
    scale_min = 5
    atg = nx.DiGraph()
    for i in range(len(cholesky)):
        if i == 0:
            atg.add_node(i,
                         C_avail=[cholesky_mapping[cholesky_name[i]]],
                         rsc_mem=defaultdict(int, {dram: 1}),
                         rsc_core=defaultdict(int, {}),
                         t_exc=defaultdict(int, {}))
        else:
            atg.add_node(i,
                         C_avail=[cholesky_mapping[cholesky_name[i]]],
                         rsc_mem=defaultdict(int, {}),
                         rsc_core=defaultdict(int, {cpu: 0.5}),
                         t_exc=defaultdict(int, {}))

    atg.add_edges_from([(i, j, {
        "comm": (cholesky[i][j] - cholesky_min) / (cholesky_max - cholesky_min) * (scale_max - scale_min) + scale_min
    }) for i in range(len(cholesky)) for j in range(len(cholesky)) if cholesky[i][j]])

    # cpu_num = 10
    # dram_num = 1
    # cholesky_solution = """x_0_10                        1.000000
    #     x_1_0                         1.000000
    #     x_2_2                         1.000000
    #     x_3_0                         1.000000
    #     x_4_3                         1.000000
    #     x_5_2                         1.000000
    #     x_6_5                         1.000000
    #     x_7_1                         1.000000
    #     x_8_6                         1.000000
    #     x_9_9                         1.000000
    #     x_10_3                        1.000000
    #     x_11_6                        1.000000
    #     x_12_5                        1.000000
    #     x_13_9                        1.000000
    #     x_14_7                        1.000000
    #     x_15_1                        1.000000
    #     x_16_7                        1.000000
    #     x_17_4                        1.000000
    #     x_18_4                        1.000000
    #     x_19_8                        1.000000
    #     x_20_8                        1.000000"""

    cpu_num = 20
    dram_num = 1
    cholesky_solution = """x_0_20                        1.000000
        x_1_0                         1.000000
        x_2_1                         1.000000
        x_3_2                         1.000000
        x_4_3                         1.000000
        x_5_4                         1.000000
        x_6_5                         1.000000
        x_7_6                         1.000000
        x_8_7                         1.000000
        x_9_8                         1.000000
        x_10_9                        1.000000
        x_11_10                        1.000000
        x_12_11                        1.000000
        x_13_12                        1.000000
        x_14_13                        1.000000
        x_15_14                        1.000000
        x_16_15                        1.000000
        x_17_16                        1.000000
        x_18_17                        1.000000
        x_19_18                        1.000000
        x_20_19                        1.000000"""

    sol = [l.split() for l in cholesky_solution.splitlines() if l]
    assignment = [[] for _ in range(cpu_num + dram_num)]
    for name, _ in sol:
        names = name.split("_")
        assignment[int(names[2])].append(int(names[1]))

    ccg = nx.DiGraph()
    pin_map = bidict()
    chiplets = []
    for i in range(cpu_num):
        for j in range(len(cpu.pins)):
            new_node = len(ccg)
            ccg.add_node(new_node)
            pin_map[new_node] = (i, j)

    for i in range(dram_num):
        for j in range(len(dram.pins)):
            new_node = len(ccg)
            ccg.add_node(new_node)
            pin_map[new_node] = (i + cpu_num, j)

    for n in ccg:
        for m in ccg:
            if m != n:
                ccg.add_edge(n, m, comm=0)
    for i in range(len(ccg)):
        if i < cpu_num:
            chiplets.append(Chiplet(name=cpu.name + "__{}".format(i), w=cpu.w, h=cpu.h, power=cpu.power, pins=cpu.pins))
        else:
            chiplets.append(Chiplet(name=dram.name + "__{}".format(i), w=dram.w, h=dram.h, power=dram.power, pins=dram.pins))
        pin_map[i] = (i, 0)
    for u, v in atg.edges:
        comm = atg[u][v]["comm"]
        idx_cpl_u = [i for i in range(len(assignment)) if u in assignment[i]][0]
        idx_cpl_v = [i for i in range(len(assignment)) if v in assignment[i]][0]
        if idx_cpl_u == idx_cpl_v:
            continue
        # cpl_u = chiplets[idx_cpl_u]
        # cpl_v = chiplets[idx_cpl_v]
        # pins_u = range(len(cpl_u.pins(angle=0)))
        # pins_v = range(len(cpl_v.pins(angle=0)))
        # prod_pins_u_v = list(itl.product(pins_u, pins_v))
        prod_pins_u_v = [(0, 0)]
        for p_u, p_v in prod_pins_u_v:
            ccg[pin_map.inverse[(idx_cpl_u, p_u)]][pin_map.inverse[(idx_cpl_v,
                                                                    p_v)]]["comm"] += math.ceil(comm / len(prod_pins_u_v))

    empty_edges = []
    for u, v, attr in ccg.edges(data=True):
        if attr["comm"] == 0:
            empty_edges.append((u, v))
    ccg.remove_edges_from(empty_edges)
    topo_graph, rtp = Sunfloor.sunfloor(dir_chaco=dir_chaco, dir_dsent=dir_dsent, task_graph=ccg, max_port=4, bw=128)
    PPA_booksim = NoC.eval_PPA_booksim(dir_booksim=dir_booksim,
                                       task_graph=ccg,
                                       topo_graph=topo_graph,
                                       rtp=rtp,
                                       cfg={
                                           "sim_cycle": 10000,
                                           "num_vcs": 4,
                                           "vc_buf_size": 4
                                       },
                                       clean=True)
    print(PPA_booksim)
    data = {"W_intp": 20, "H_intp": 20, "chiplets": chiplets, "tile_size": tile_size, "task_graph": ccg, "pin_map": pin_map}
    return data


if __name__ == "__main__":
    dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    dir_cplex = os.path.join(dir_root, "tool/cplex")
    dir_dataset = os.path.join(dir_root, "data/dataset")
    dir_chaco = os.path.join(dir_root, "tool/chaco")
    dir_dsent = os.path.join(dir_root, "tool/dsent")
    dir_booksim = os.path.join(dir_root, "tool/booksim")

    cpu = CType(
        name="CPU",  # Waferscale
        w=4,
        h=3,
        power=0.175 * (4 * 3) / (3.15 * 2.4),
        cost=4 * 3,
        pins=[(0, 0)],  # only consider L3-EXT-MEM
        bw_pin=128,
        rsc_core=14,
        rsc_mem=0)
    dsp = CType(name="DSP",
                w=3,
                h=3,
                power=0.5 * (3 * 3) / (2.5 * 2.5),
                cost=3 * 3,
                pins=[(0, 0), (2, 2)],
                bw_pin=128,
                rsc_core=1,
                rsc_mem=0)
    dram = CType(name="DRAM",
                 w=9,
                 h=9,
                 power=20 * (9 * 9) / (8.75 * 8.75),
                 cost=9 * 9,
                 pins=[(0, 0)],
                 bw_pin=128,
                 rsc_core=0,
                 rsc_mem=2)
    tile_size = 1e-3

    dt = [app1(), app2()]
    with open(os.path.join(dir_dataset, "sys_{}_{}.pkl".format(0, 0)), "wb") as outp:
        pickle.dump(dt, outp, pickle.HIGHEST_PROTOCOL)