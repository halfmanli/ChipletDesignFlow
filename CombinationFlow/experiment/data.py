import pickle
import os
import math
import numpy as np
from rectpack import newPacker
from framework.ChipletSys import ChipletSys
from framework.NoC.NoC import eval_router_dsent

groups = [(6, 10), (11, 15), (16, 20)]
groups_real_app = [(0, 0)]
dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
dir_hotspot = os.path.join(dir_root, "tool/hotspot")
dir_dsent = os.path.join(dir_root, "tool/dsent")
dir_dataset = os.path.join(dir_root, "data/dataset")
dir_ActiveGIA = os.path.join(dir_root, "data/ActiveGIA")
dir_SISL_mesh = os.path.join(dir_root, "data/SISL/mesh")
dir_SISL_ft = os.path.join(dir_root, "data/SISL/ft")
dir_SISL_bd = os.path.join(dir_root, "data/SISL/bd")
dir_PassiveGIA = os.path.join(dir_root, "data/PassiveGIA")
dir_SiP = os.path.join(dir_root, "data/SiP")
dir_ideal_active = os.path.join(dir_root, "data/SISL/app_active")
dir_ideal_passive = os.path.join(dir_root, "data/SISL/app_passive")


def compare(dir_a, dir_b, avg=True, real_app=False):
    """
        return power_a / power_b, perf_a / per_b
    """
    power_a_b = {}
    perf_a_b = {}
    g = groups_real_app if real_app else groups
    for (l, h) in g:
        power_a_b[(l, h)] = []
        perf_a_b[(l, h)] = []
        for idx_d in range(50):
            power_a = math.inf
            power_b = math.inf
            perf_a = math.inf
            perf_b = math.inf
            for pid in range(2):
                path_a = os.path.join(dir_a, "log_{}_{}".format(l, h), "{}/{}.pkl".format(idx_d, pid))
                path_b = os.path.join(dir_b, "log_{}_{}".format(l, h), "{}/{}.pkl".format(idx_d, pid))

                if os.path.exists(path_a):
                    with open(path_a, "rb") as f:
                        data_a = pickle.load(f)
                        if power_a + perf_a > data_a[2]["power_eu"] + data_a[2]["perf"]:
                            power_a = data_a[2]["power_eu"]
                            perf_a = data_a[2]["perf"]

                if os.path.exists(path_b):
                    with open(path_b, "rb") as f:
                        data_b = pickle.load(f)
                        if power_b + perf_b > data_b[2]["power_eu"] + data_b[2]["perf"]:
                            power_b = data_b[2]["power_eu"]
                            perf_b = data_b[2]["perf"]

            if math.isinf(power_a) or math.isinf(power_b) or math.isinf(perf_a) or math.isinf(perf_b):
                continue
            power_a_b[(l, h)].append(power_a / power_b)
            perf_a_b[(l, h)].append(perf_a / perf_b)
        if avg:
            power_a_b[(l, h)] = np.average(power_a_b[(l, h)])
            perf_a_b[(l, h)] = np.average(perf_a_b[(l, h)])
    return {"power": power_a_b, "perf": perf_a_b}


def extract(dir_data, real_app=False):
    power = {}
    perf = {}
    thermal = {}
    g = groups_real_app if real_app else groups
    for (l, h) in g:
        power[(l, h)] = []
        perf[(l, h)] = []
        thermal[(l, h)] = []
        for idx_d in range(50):
            power_data = math.inf
            perf_data = math.inf
            thermal_data = math.inf
            for pid in range(2):
                path_data = os.path.join(dir_data, "log_{}_{}".format(l, h), "{}/{}.pkl".format(idx_d, pid))

                if os.path.exists(path_data):
                    with open(path_data, "rb") as f:
                        data = pickle.load(f)
                        if power_data + perf_data > data[2]["power_eu"] + data[2]["perf"]:
                            power_data = data[2]["power_eu"]
                            perf_data = data[2]["perf"]
                            thermal_data = data[2]["thermal"]

            if math.isinf(power_data) or math.isinf(perf_data):
                continue
            power[(l, h)].append(power_data)
            perf[(l, h)].append(perf_data)
            thermal[(l, h)].append(thermal_data)
    if real_app:
        return power, perf, thermal
    else:
        return power, perf


def get_area():
    area = {}
    for (l, h) in groups:
        path_dt = os.path.join(dir_dataset, "sys_{}_{}_topt.pkl".format(l, h))
        dt = pickle.load(open(path_dt, "rb"))
        area[(l, h)] = min(sum([c.w_orig * c.h_orig for c in d["chiplets"]]) / (d["W_intp"] * d["H_intp"])
                           for d in dt[:50])  # only consider 50 applications
    return area


def get_power():
    """
        return minimum power of each scale of system
    """
    power = {}
    for (l, h) in groups:
        path_dt = os.path.join(dir_dataset, "sys_{}_{}_topt.pkl".format(l, h))
        dt = pickle.load(open(path_dt, "rb"))
        power[(l, h)] = min([sum([c.power for c in d["chiplets"]]) for d in dt[:50]])
    return power


def get_thermal():
    therm_impr = {}
    for (low, high) in groups:
        therm_impr[(low, high)] = []
        path_dt = os.path.join(dir_dataset, "sys_{}_{}_topt.pkl".format(low, high))
        dt = pickle.load(open(path_dt, "rb"))[:50]
        for d in dt:
            csys = ChipletSys(W=d["W_intp"],
                              H=d["H_intp"],
                              chiplets=d["chiplets"],
                              task_graph=d["task_graph"],
                              pin_map=d["pin_map"])

            packer = newPacker()
            for idx_cpl, cpl in enumerate(csys.chiplets):
                packer.add_rect(width=cpl.w_orig, height=cpl.h_orig, rid=idx_cpl)
            packer.add_bin(width=csys.W, height=csys.H, count=1)
            packer.pack()
            all_rects = packer.rect_list()
            assert len(all_rects) == len(csys.chiplets)
            init_pl = [0] * len(csys.chiplets)  # initial placement
            for rect in all_rects:
                _, x, y, w, h, rid = rect
                assert (w == csys.chiplets[rid].w_orig
                        and h == csys.chiplets[rid].h_orig) or (h == csys.chiplets[rid].w_orig
                                                                and w == csys.chiplets[rid].h_orig)
                angle = 0 if (w == csys.chiplets[rid].w_orig and h == csys.chiplets[rid].h_orig) else 1
                init_pl[rid] = (x, y, angle)

            therm_before = csys.eval_thermal(dir_hotspot=dir_hotspot,
                                             tile_size=d["tile_size"],
                                             placement=init_pl,
                                             visualize=False).max()
            therm_after = csys.eval_thermal(dir_hotspot=dir_hotspot,
                                            tile_size=d["tile_size"],
                                            placement=d["placement"][0],
                                            visualize=False).max()
            therm_impr[(low, high)].append(therm_before - therm_after)
        therm_impr[(low, high)] = (np.average(therm_impr[(low, high)]), np.max(therm_impr[(low, high)]))
    return therm_impr


def get_yield(A, D, alpha):
    return (1 + A * D / alpha)**(-alpha)


def get_hop():
    avg_hop = {}
    for (low, high) in groups:
        avg_hop[(low, high)] = []
        path_dt = os.path.join(dir_dataset, "sys_{}_{}_topt.pkl".format(low, high))
        dt = pickle.load(open(path_dt, "rb"))[:50]
        for d in dt:
            avg_hop[(low, high)] += [len(p) + 1 for p in d["rtp"].values()]
        avg_hop[(low, high)] = np.average(avg_hop[(low, high)])
    return avg_hop


def get_percent_cycle(cycle):
    percent_one_cycle = {}
    for (low, high) in groups:
        percent_one_cycle[(low, high)] = []
        for idx_d in range(50):
            path_data = os.path.join(dir_PassiveGIA, "log_{}_{}/{}/0.pkl".format(low, high, idx_d))
            data = pickle.load(open(path_data, "rb"))
            paths = data[2]["mlayout"][1]
            percent_one_cycle[(low, high)].append(len([p for p in paths.values() if len(p) <= 7 * cycle]) / len(paths))
        percent_one_cycle[(low, high)] = np.average(percent_one_cycle[(low, high)])
    return percent_one_cycle


# active GIA vs. SISL
print("*" * 100)
print("active GIA vs. SISL")
ma = compare(dir_SISL_mesh, dir_ActiveGIA)
fa = compare(dir_SISL_ft, dir_ActiveGIA)
ba = compare(dir_SISL_bd, dir_ActiveGIA)
print("SISL mesh: ", ma)
print("SISL ft: ", fa)
print("SISL bd: ", ba)
power_ma, perf_ma = ma["power"], ma["perf"]
power_fa, perf_fa = fa["power"], fa["perf"]
power_ba, perf_ba = ba["power"], ba["perf"]
print("active GIA over SISL: ", "power:",
      np.average(list(power_ma.values()) + list(power_fa.values()) + list(power_ba.values())), " perf:",
      np.average(list(perf_ma.values()) + list(perf_fa.values()) + list(perf_ba.values())))
print("*" * 100)

# passive GIA vs. SiP
print("passive GIA vs. SiP")
perf_SiP = extract(dir_SiP)[1]
# perf_SiP = dict([(k, min(zip(v, range(len(v))))) for k, v in perf_SiP.items()])
print([(np.array(v) > 1000).sum() / len(v) for k, v in perf_SiP.items()])
sp = compare(dir_SiP, dir_PassiveGIA)
power_sp, perf_sp = sp["power"], sp["perf"]
print("passive GIA over SiPterposer: ", "power:",
      np.average(list(power_sp.values())), " perf:",
      np.average(list(perf_sp.values())))
print(sp)

print("*" * 100)
print("overhead of GIA")
print("active GIA overhead compared with custom active: ", compare(dir_ActiveGIA, dir_ideal_active))
print(compare(dir_ideal_passive, dir_ideal_active))
print(compare(dir_PassiveGIA, dir_ideal_active))
print("passive GIA overhead compared with custom passive: ", compare(dir_PassiveGIA, dir_ideal_passive))
print(get_area())
# print(get_thermal())

print("*" * 100)
print("overhead of GIA router")
router_5_5 = eval_router_dsent(dir_dsent=dir_dsent,
                               in_port=5,
                               out_port=5,
                               load=0.5,
                               cfg={
                                   "process": 45,
                                   "freq": 1e9,
                                   "channel_width": 128,
                                   "num_vc": 4,
                                   "vc_buf_size": 4
                               })
router_4_4 = eval_router_dsent(dir_dsent=dir_dsent,
                               in_port=4,
                               out_port=4,
                               load=0.5,
                               cfg={
                                   "process": 45,
                                   "freq": 1e9,
                                   "channel_width": 128,
                                   "num_vc": 4,
                                   "vc_buf_size": 4
                               })
router_1_1 = eval_router_dsent(dir_dsent=dir_dsent,
                               in_port=1,
                               out_port=1,
                               load=0.5,
                               cfg={
                                   "process": 45,
                                   "freq": 1e9,
                                   "channel_width": 128,
                                   "num_vc": 1,
                                   "vc_buf_size": 128
                               })
router_8_8 = eval_router_dsent(dir_dsent=dir_dsent,
                               in_port=8,
                               out_port=8,
                               load=0.5,
                               cfg={
                                   "process": 45,
                                   "freq": 1e9,
                                   "channel_width": 128,
                                   "num_vc": 1,
                                   "vc_buf_size": 1
                               })
area_8_8 = router_4_4["total_area"] - router_4_4["xbar_area"] + router_8_8["xbar_area"] + 4 * router_1_1["buf_area"] / 128
print("area: ", area_8_8, router_5_5["total_area"], "ratio:", area_8_8 / router_5_5["total_area"])
power_8_8 = router_4_4["total_power"] - (router_4_4["xbar_dynamic"] + router_4_4["xbar_leakage"]) + (
    router_8_8["xbar_dynamic"] +
    router_8_8["xbar_leakage"]) + 4 * (router_1_1["buffer_dynamic"] + router_1_1["buffer_leakage"]) / 128
print("power: ", power_8_8 / router_5_5["total_power"])
print("yield: ", "5_5: ",
      get_yield(40 * 40 * (router_5_5["total_area"] / 1e-6), 2000 / 1e6, 3) * 100, "8_8: ",
      get_yield(40 * 40 * (area_8_8 / 1e-6), 2000 / 1e6, 3) * 100)

power_active_GIA = extract(dir_ActiveGIA)[0]
power_passive_GIA = extract(dir_PassiveGIA)[0]
print("max power: ", "active GIA ", [max(v) for v in power_active_GIA.values()], ";passive GIA ",
      [max(v) for v in power_passive_GIA.values()])
print("network topology average hops: ", get_hop())
print("percentage of paths can be traversed by passive GIA in one cycle: ", get_percent_cycle(1), get_percent_cycle(3))

print("*" * 100)
print("real applications")
print("active GIA data: ", extract(dir_ActiveGIA, real_app=True))
print("SISL mesh data: ", extract(dir_SISL_mesh, real_app=True))
print("SISL ft data: ", extract(dir_SISL_ft, real_app=True))
print("SISL bd data: ", extract(dir_SISL_bd, real_app=True))
print("passive GIA data: ", extract(dir_PassiveGIA, real_app=True))
print("SiPterposer data: ", extract(dir_SiP, real_app=True))

print("*" * 100)
print("minimum power of each scale of system: ", get_power())