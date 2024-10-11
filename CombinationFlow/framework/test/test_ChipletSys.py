from ..ChipletSys import Chiplet, ChipletSys, gen_csys
import argparse
import numpy as np
import os

dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
dir_tgff = os.path.join(dir_root, "tool/tgff")
dir_hotspot = os.path.join(dir_root, "tool/hotspot")


def test_chiplet():
    cpl = Chiplet(name="test", w=4, h=3, power=40, pins=[(0, 0), (1, 1), (1, 0), (3, 2)])
    cpl.show(0)
    cpl.show(1)
    cpl.show(2)
    cpl.show(3)


def test_hotspot():
    # Case Study 1: Multi-GPU System
    c_GPU0 = Chiplet(name="GPU0", w=18, h=18, power=295, pins=[(0, 0)])
    c_HBM1 = Chiplet(name="HBM1", w=8, h=12, power=20.87, pins=[(0, 0)])
    c_HBM2 = Chiplet(name="HBM2", w=8, h=12, power=20.87, pins=[(0, 0)])
    c_HBM0 = Chiplet(name="HBM0", w=8, h=12, power=20.87, pins=[(0, 0)])
    c_CPU = Chiplet(name="CPU", w=12, h=12, power=105, pins=[(0, 0)])
    c_GPU1 = Chiplet(name="GPU1", w=18, h=18, power=288.55, pins=[(0, 0)])
    csys = ChipletSys(45, 45, [c_GPU0, c_HBM1, c_HBM2, c_HBM0, c_CPU, c_GPU1])

    placement = [(1, 21, 0), (21, 27, 0), (31, 27, 0), (1, 6, 0), (11, 6, 0), (25, 6, 0)]
    # csys.show_placement(placement=placement)
    print(
        "paper: 95.31, mine:",
        csys.eval_thermal(dir_hotspot=dir_hotspot,
                          tile_size=0.001,
                          placement=placement,
                          visualize=False,
                          cfg={
                              "W_tile": csys.W,
                              "H_tile": csys.H,
                              "ubump_dia": 0.000010,
                              "ubump_pitch": 0.000020
                          }).max())

    placement = [(0, 26, 0), (20, 35, 1), (16, 1, 0), (1, 15, 1), (1, 1, 0), (25, 1, 0)]
    csys.show_placement(placement=placement)
    print(
        "paper: 91.25, mine:",
        csys.eval_thermal(dir_hotspot=dir_hotspot,
                          tile_size=0.001,
                          placement=placement,
                          visualize=True,
                          cfg={
                              "W_tile": csys.W,
                              "H_tile": csys.H,
                              "ubump_dia": 0.000010,
                              "ubump_pitch": 0.000020
                          }).max())

    placement = [(26, 26, 0), (16, 31, 0), (20, 6, 0), (9, 22, 1), (2, 31, 1), (1, 1, 0)]
    csys.show_placement(placement=placement)
    print(
        "paper: 91.52, mine:",
        csys.eval_thermal(dir_hotspot=dir_hotspot,
                          tile_size=0.001,
                          placement=placement,
                          visualize=True,
                          cfg={
                              "W_tile": csys.W,
                              "H_tile": csys.H,
                              "ubump_dia": 0.000010,
                              "ubump_pitch": 0.000020
                          }).max())

    # Case Study 2: CPU-DRAM System
    c_CPU0 = Chiplet(name="CPU0", w=8, h=9, power=145.45, pins=[(0, 0)])
    c_CPU1 = Chiplet(name="CPU1", w=8, h=9, power=145.45, pins=[(0, 0)])
    c_CPU2 = Chiplet(name="CPU2", w=8, h=9, power=145.45, pins=[(0, 0)])
    c_CPU3 = Chiplet(name="CPU3", w=8, h=9, power=145.45, pins=[(0, 0)])

    c_DRAM0 = Chiplet(name="DRAM0", w=9, h=9, power=18.9, pins=[(0, 0)])
    c_DRAM1 = Chiplet(name="DRAM1", w=9, h=9, power=18.9, pins=[(0, 0)])
    c_DRAM2 = Chiplet(name="DRAM2", w=9, h=9, power=18.9, pins=[(0, 0)])
    c_DRAM3 = Chiplet(name="DRAM3", w=9, h=9, power=18.9, pins=[(0, 0)])

    csys = ChipletSys(45, 45, [c_CPU0, c_CPU1, c_CPU2, c_CPU3, c_DRAM0, c_DRAM1, c_DRAM2, c_DRAM3])
    placement = [(15, 13, 0), (25, 13, 0), (25, 23, 0), (15, 23, 0), (4, 13, 0), (35, 13, 0), (4, 23, 0), (35, 23, 0)]
    csys.show_placement(placement=placement)
    print(
        "paper: 115.94, mine:",
        csys.eval_thermal(dir_hotspot=dir_hotspot,
                          tile_size=0.001,
                          placement=placement,
                          visualize=True,
                          cfg={
                              "W_tile": csys.W,
                              "H_tile": csys.H,
                              "ubump_dia": 0.000010,
                              "ubump_pitch": 0.000020
                          }).max())

    placement = [(6, 29, 0), (6, 17, 0), (18, 5, 0), (16, 17, 0), (16, 29, 0), (6, 5, 0), (28, 5, 0), (26, 17, 0)]
    csys.show_placement(placement=placement)
    print(
        "paper: 113.54, mine:",
        csys.eval_thermal(dir_hotspot=dir_hotspot,
                          tile_size=0.001,
                          placement=placement,
                          visualize=True,
                          cfg={
                              "W_tile": csys.W,
                              "H_tile": csys.H,
                              "ubump_dia": 0.000010,
                              "ubump_pitch": 0.000020
                          }).max())

    # Case Study 3: Huawei Ascend 910 System
    c_HBM0 = Chiplet(name="HBM0", w=8, h=12, power=20.87, pins=[(0, 0)])
    c_HBM1 = Chiplet(name="HBM1", w=8, h=12, power=20.87, pins=[(0, 0)])
    c_HBM2 = Chiplet(name="HBM2", w=8, h=12, power=20.87, pins=[(0, 0)])
    c_HBM3 = Chiplet(name="HBM3", w=8, h=12, power=20.87, pins=[(0, 0)])
    c_Virtuvian = Chiplet(name="Virtuvian", w=31, h=14, power=244, pins=[(0, 0)])
    c_Nimbus = Chiplet(name="Nimbus", w=10, h=16, power=13.33, pins=[(0, 0)])
    csys = ChipletSys(45, 45, [c_HBM0, c_HBM1, c_HBM2, c_HBM3, c_Virtuvian, c_Nimbus])
    placement = [(0, 30, 1), (16, 30, 1), (0, 7, 1), (16, 7, 1), (0, 16, 0), (33, 15, 0)]
    csys.show_placement(placement=placement)
    print(
        "paper: 75.48, mine:",
        csys.eval_thermal(dir_hotspot=dir_hotspot,
                          tile_size=0.001,
                          placement=placement,
                          visualize=True,
                          cfg={
                              "W_tile": csys.W,
                              "H_tile": csys.H,
                              "ubump_dia": 0.000010,
                              "ubump_pitch": 0.000020
                          }).max())


def test_gen_csys():
    csys = gen_csys(dir_tgff=dir_tgff,
                    bw=128,
                    cfg_tgff={
                        "task_cnt": (50, 5),
                        "seed": 0,
                        "tg_cnt": 1000
                    },
                    filter=None,
                    clean=False,
                    cfg_csys={
                        "W_intp": 50,
                        "H_intp": 50,
                        "max_size_cpl": 20,
                        "max_wh_ratio_cpl": 2,
                        "max_area_sys": 0.5 * 50 * 50,
                        "max_num_cpl": 20,
                        "min_num_cpl": 10,
                        "max_pd_cpl": 0.8,
                        "min_pd_cpl": 0.2,
                        "max_pd_sys": 0.5,
                        "min_pd_sys": 0.38,
                        "max_try": 50
                    })
    csys[100].show_CCG()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tests for ChipletSys package.")
    parser.add_argument("test_func")
    args = parser.parse_args()
    eval((args.test_func) + "()")