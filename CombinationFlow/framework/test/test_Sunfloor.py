from ..Sunfloor import *
import numpy as np
import argparse
from ..NoC import gen_task_tgff, eval_PPA_booksim
from tqdm import tqdm, trange
import os

dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
dir_dsent = os.path.join(dir_root, "tool/dsent")
dir_tgff = os.path.join(dir_root, "tool/tgff")
dir_chaco = os.path.join(dir_root, "tool/chaco")
dir_booksim = os.path.join(dir_root, "tool/booksim")


def test_sunfloor():
    cfg = {"task_cnt": (20, 15), "seed": np.random.randint(1e9), "tg_cnt": 100}
    gs = gen_task_tgff(dir_tgff=dir_tgff, bw=128, cfg=cfg, clean=True)
    for task_graph in tqdm(gs):
        topo_graph, rtp = sunfloor(dir_chaco=dir_chaco, dir_dsent=dir_dsent, task_graph=task_graph, max_port=4, bw=128)
        PPA = eval_PPA_booksim(dir_booksim=dir_booksim,
                               task_graph=task_graph,
                               topo_graph=topo_graph,
                               rtp=rtp,
                               cfg={
                                   "sim_cycle": 100000,
                                   "num_vcs": 4,
                                   "vc_buf_size": 4
                               })
        print(PPA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tests for Sunfloor package.")
    parser.add_argument("test_func")
    args = parser.parse_args()
    eval((args.test_func) + "()")