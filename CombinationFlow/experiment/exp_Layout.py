import pickle
from multiprocessing import Pool
import matplotlib as mpl
from matplotlib import pyplot as plt
from framework import NoC, ChipletSys, GIA, Sunfloor
from rectpack import newPacker
import os
import numpy as np
import pandas as pd
import seaborn as sns


def gen_topo_dt_single(nnum):
    while True:
        tgs = NoC.gen_task_tgff(dir_tgff=dir_tgff,
                                bw=128,
                                cfg={
                                    "tg_cnt": 50,
                                    "task_cnt": (nnum, 1),
                                    "seed": np.random.randint(1e9)
                                })
        task_graph = None
        for tg in tgs:
            if len(tg) == nnum:
                task_graph = tg
                break
        if task_graph:
            break
    topo_graph, rtp = Sunfloor.sunfloor(dir_chaco=dir_chaco, dir_dsent=dir_dsent, task_graph=task_graph, max_port=4, bw=128)
    return task_graph, topo_graph, rtp


def gen_topo_dt(min_nnum, max_nnum, num_tg):
    """
        min_nnum/max_nnum: minimum/maximum number of cores in task graph
        num_tg: number of task graphs per nnum
        generate NoC topology dataset
    """
    num_cpu = 32
    pool = Pool(processes=num_cpu)
    res = []
    for nnum in range(min_nnum, max_nnum + 1):
        for _ in range(num_tg):
            res.append(pool.apply_async(gen_topo_dt_single, args=(nnum, )))
    pool.close()
    res_dump = []
    for r in res:
        res_dump.append(r.get())
        with open(os.path.join(dir_dt, "topo.pkl"), "wb") as outp:
            pickle.dump(res_dump, outp, pickle.HIGHEST_PROTOCOL)


def gen_layout_single(type_channel, task_graph, topo_graph, rtp):
    nnum = len(task_graph)
    chiplets = [ChipletSys.Chiplet(name=str(idx_n), w=1, h=1, power=1, pins=[(0, 0)]) for idx_n in range(nnum)]
    csys = ChipletSys.ChipletSys(W=W_intp,
                                 H=H_intp,
                                 chiplets=chiplets,
                                 task_graph=task_graph,
                                 pin_map=dict([(n, (n, 0)) for n in range(nnum)]))

    # generate initial placement
    packer = newPacker()
    for idx_cpl, cpl in enumerate(csys.chiplets):
        packer.add_rect(width=cpl.w_orig, height=cpl.h_orig, rid=idx_cpl)
    packer.add_bin(width=csys.W, height=csys.H, count=1)
    packer.pack()
    all_rects = packer.rect_list()
    assert len(all_rects) == len(csys.chiplets)
    pl = [0] * len(csys.chiplets)
    for rect in all_rects:
        _, x, y, w, h, rid = rect
        assert (w == csys.chiplets[rid].w_orig and h == csys.chiplets[rid].h_orig) or (h == csys.chiplets[rid].w_orig
                                                                                       and w == csys.chiplets[rid].h_orig)
        angle = 0 if (w == csys.chiplets[rid].w_orig and h == csys.chiplets[rid].h_orig) else 1
        pl[rid] = (x, y, angle)

    placer = GIA.PlacerSA(csys=csys, cost_func=None, dir_log=None)
    for _ in range(50):  # scatter the compact placement
        new_pl = placer.neighbor(placement=pl, weight_act=[0.25, 0.25, 0.25, 0.25], max_move=4)
        if new_pl:
            pl = new_pl

    total_try = 50
    cnt_suc = 0
    for _ in range(total_try):
        if type_channel == "dual channel":
            flg_suc, _, __ = GIA.gen_layout_active(csys=csys, topo_graph=topo_graph, rtp=rtp, placement=pl, max_retry=90)
        else:
            flg_suc, _, __ = GIA.gen_layout_active_sc(csys=csys, topo_graph=topo_graph, rtp=rtp, placement=pl, max_retry=90)
        if flg_suc:
            cnt_suc += 1
        new_pl = placer.neighbor(placement=pl, weight_act=[0.25, 0.25, 0.25, 0.25], max_move=4)
        if new_pl:
            pl = new_pl

    return type_channel, nnum, total_try, cnt_suc


def gen_layout():
    mpl.use('agg')
    num_cpu = 32
    pool = Pool(processes=num_cpu)
    res = []
    dt = pickle.load(open(os.path.join(dir_dt, "topo.pkl"), "rb"))
    for idx_d, (task_graph, topo_graph, rtp) in enumerate(dt):
        res.append(pool.apply_async(gen_layout_single, args=("single channel", task_graph, topo_graph, rtp)))
        res.append(pool.apply_async(gen_layout_single, args=("dual channel", task_graph, topo_graph, rtp)))
    pool.close()
    hist_raw = {"dual channel": {}, "single channel": {}}
    for r in res:
        type_channel, nnum, total_try, cnt_suc = r.get()
        print(nnum, total_try, cnt_suc)
        if nnum not in hist_raw[type_channel]:
            hist_raw[type_channel][nnum] = (0, 0)
        total_try_old, cnt_suc_old = hist_raw[type_channel][nnum]
        hist_raw[type_channel][nnum] = total_try_old + total_try, cnt_suc_old + cnt_suc
        print(hist_raw)
        data = []
        for k, (t, c) in hist_raw["dual channel"].items():
            data.append(("dual channel", k, c / t * 100))
        for k, (t, c) in hist_raw["single channel"].items():
            data.append(("single channel", k, c / t * 100))

        df = pd.DataFrame(data, columns=["type_channel", "scale", "suc"])
        g = sns.lineplot(data=df, x="scale", y="suc", hue="type_channel")
        g.legend().set_title(None)
        plt.savefig("exp_Layout.png")
        plt.close()


if __name__ == "__main__":
    dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    dir_tgff = os.path.join(dir_root, "tool/tgff")
    dir_dsent = os.path.join(dir_root, "tool/dsent")
    dir_chaco = os.path.join(dir_root, "tool/chaco")
    dir_dt = os.path.join(dir_root, "data/dataset")
    W_intp = 30
    H_intp = W_intp
    # gen_topo_dt(min_nnum=10, max_nnum=100, num_tg=10)
    gen_layout()
