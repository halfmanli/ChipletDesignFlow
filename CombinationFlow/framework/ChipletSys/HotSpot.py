import subprocess
import tempfile
import shutil
from os import path
from .utils import is_overlap_with_rects, is_rects_overlap
import numpy as np


def fill_space(W, H, rects):
    """
        the hotspot require all area covered by small rectangles, split the region not covered by chiplet to rectangle
        Return: rectangle regions exclude rects

        W, H: the width and height of target region
        rects: list of coordinates, (x_bl, y_bl, x_ur, y_ur)
    """
    interval_x = [0, W] + [x for (x_bl, _, x_ur, _) in rects for x in (x_bl, x_ur)]  # (left ~ right)
    interval_x = set(interval_x)  # remove duplicated numbers
    interval_x = list(sorted(interval_x))
    # zip is iterable, need to be converted
    interval_x = list(zip(interval_x[:-1], interval_x[1:]))

    interval_y = [0, H] + [y for (_, y_bl, _, y_ur) in rects for y in (y_bl, y_ur)]
    interval_y = set(interval_y)  # remove duplicated numbers
    interval_y = list(sorted(interval_y))
    interval_y = list(zip(interval_y[:-1], interval_y[1:]))

    space_rects = []

    for (x_0, x_1) in interval_x:  # generate the empty zone first; remove the rectangle inside other rectangles
        for (y_0, y_1) in interval_y:
            if not is_overlap_with_rects(x_0, y_0, x_1, y_1, rects):
                space_rects.append((x_0, y_0, x_1, y_1))

    if True:  # check
        assert sum([(x_ur - x_bl) * (y_ur - y_bl) for (x_bl, y_bl, x_ur, y_ur) in rects + space_rects]) == W * H
        assert not is_rects_overlap(rects + space_rects)

    return space_rects


def get_thermal_hotspot(chiplet_comp, dir_hotspot, clean=True, config=None):
    """
        chiplet_comp: list of dict. chiplet component dict.field: width, height, left_x, bottom_y, power, name
        dir_hotspot: dir of hotspot and dir_hotspot/build is the directory for storing intermediate files
        clean: clean temporary files
    """
    si_sheat = 1.75e6  # volumetric heat capacity
    si_resist = 0.01
    cu_sheat = 3494400
    cu_resist = 0.0025
    uf_sheat = 2320000
    uf_resist = 0.625

    if config is None:
        config = {}
    W = config["W_tile"]  # if "W" in config else 50  # interposer width, unit: tile
    H = config["H_tile"]  # if "H" in config else 50  # interposer height
    size_tile = config["size_tile"] if "size_tile" in config else 0.001
    size_grid = config["size_grid"] if "size_grid" in config else 64

    c4_dia = config["c4_dia"] if "c4_dia" in config else 0.000250  # 250um
    c4_pitch = config["c4_pitch"] if "c4_pitch" in config else 0.000600  # 600um
    TSV_dia = config["TSV_dia"] if "TSV_dia" in config else 0.000010  # 10um
    TSV_pitch = config["TSV_pitch"] if "TSV_pitch" in config else 0.000050  # 50um
    ubump_dia = config["ubump_dia"]  if "ubump_dia" in config else 0.000010  # 10um
    ubump_pitch = config["ubump_pitch"]  if "ubump_pitch" in config else 0.000020  # 20um

    aratio_c4 = (c4_pitch / c4_dia) * (c4_pitch / c4_dia) - 1  # ratio of white area and c4 area
    aratio_TSV = (TSV_pitch / TSV_dia) * (TSV_pitch / TSV_dia) - 1
    aratio_ubump = (ubump_pitch / ubump_dia) * (ubump_pitch / ubump_dia) - 1

    c4_resist = (1 + aratio_c4) * cu_resist * uf_resist / (uf_resist + aratio_c4 * cu_resist)
    TSV_resist = (1 + aratio_TSV) * cu_resist * si_resist / (si_resist + aratio_TSV * cu_resist)
    ubump_resist = (1 + aratio_ubump) * cu_resist * uf_resist / (uf_resist + aratio_ubump * cu_resist)

    c4_sheat = (cu_sheat + aratio_c4 * uf_sheat) / (1 + aratio_c4)
    TSV_sheat = (cu_sheat + aratio_TSV * si_sheat) / (1 + aratio_TSV)
    ubump_sheat = (cu_sheat + aratio_ubump * uf_sheat) / (1 + aratio_ubump)

    mat_c4 = "\t" + str(c4_sheat) + "\t" + str(c4_resist) + "\n"
    mat_TSV = "\t" + str(TSV_sheat) + "\t" + str(TSV_resist) + "\n"
    mat_ubump = "\t" + str(ubump_sheat) + "\t" + str(ubump_resist) + "\n"
    mat_si = "\t" + str(si_sheat) + "\t" + str(si_resist) + "\n"
    mat_uf = "\t" + str(uf_sheat) + "\t" + str(uf_resist) + "\n"

    chiplet_rects = [(cp["left_x"], cp["bottom_y"], cp["left_x"] + cp["width"], cp["bottom_y"] + cp["height"])
                     for cp in chiplet_comp]  # field: width, height, left_x, bottom_y, power, name
    space_rects = fill_space(W, H, chiplet_rects)

    dir_tmp = tempfile.mkdtemp(dir=path.join(dir_hotspot, "build"))  # sub dir

    L0_Substrate_path = path.join(dir_tmp, "L0_Substrate.flp")
    L1_C4Layer_path = path.join(dir_tmp, "L1_C4Layer.flp")
    L2_Interposer_path = path.join(dir_tmp, "L2_Interposer.flp")
    L3_UbumpLayer_path = path.join(dir_tmp, "L3_UbumpLayer.flp")
    L4_ChipLayer_path = path.join(dir_tmp, "L4_ChipLayer.flp")
    L4_Ptrace_path = path.join(dir_tmp, "L4.ptrace")
    L5_TIM_path = path.join(dir_tmp, "L5_TIM.flp")
    Config_in_path = path.join(dir_hotspot, "hotspot.config.template")
    Config_out_path = path.join(dir_tmp, "hotspot.config")
    LCF_path = path.join(dir_tmp, "layers.lcf")
    Steady_path = path.join(dir_tmp, "res.steady")
    GridSteady_path = path.join(dir_tmp, "res.grid.steady")

    with open(L0_Substrate_path, "w") as L0_Substrate:
        L0_Substrate.write("# Floorplan for Substrate Layer with width " + str(W * size_tile) + "x height" +
                           str(H * size_tile) + " m\n")
        L0_Substrate.write("Substrate\t" + str(W * size_tile) + "\t" + str(H * size_tile) + "\t0.0\t0.0\n")

    with open(L1_C4Layer_path, "w") as L1_C4Layer:
        L1_C4Layer.write("# Floorplan for c4 Layer \n")
        L1_C4Layer.write("C4Layer\t" + str(W * size_tile) + "\t" + str(H * size_tile) + "\t0.0\t0.0" + mat_c4)

    with open(L2_Interposer_path, "w") as L2_Interposer:
        L2_Interposer.write("# Floorplan for Silicon Interposer Layer\n")
        L2_Interposer.write("Interposer\t" + str(W * size_tile) + "\t" + str(H * size_tile) + "\t0.0\t0.0" + mat_TSV)

    with open(L3_UbumpLayer_path, "w") as L3_UbumpLayer, open(L4_ChipLayer_path, "w") as L4_ChipLayer:
        L3_UbumpLayer.write("# Floorplan for Microbump Layer \n")
        L4_ChipLayer.write("# Floorplan for Chip Layer\n")

        for idx_cp, cp in enumerate(chiplet_comp):
            L3_UbumpLayer.write("Ubump_" + str(idx_cp) + "\t" + str(cp["width"] * size_tile) + "\t" +
                                str(cp["height"] * size_tile) + "\t" + str(cp["left_x"] * size_tile) + "\t" +
                                str(cp["bottom_y"] * size_tile) + mat_ubump)  # place ubump under chiplet
            L4_ChipLayer.write(cp["name"] + "\t" + str(cp["width"] * size_tile) + "\t" + str(cp["height"] * size_tile) + "\t" +
                               str(cp["left_x"] * size_tile) + "\t" + str(cp["bottom_y"] * size_tile) +
                               mat_si)  # place silicon on chiplet layer

        for idx_rs, (x_bl, y_bl, x_ur, y_ur) in enumerate(space_rects):
            L3_UbumpLayer.write("Empty_" + str(idx_rs) + "\t" + str((x_ur - x_bl) * size_tile) + "\t" +
                                str((y_ur - y_bl) * size_tile) + "\t" + str(x_bl * size_tile) + "\t" + str(y_bl * size_tile) +
                                mat_uf)
            L4_ChipLayer.write("Empty_" + str(idx_rs) + "\t" + str((x_ur - x_bl) * size_tile) + "\t" +
                               str((y_ur - y_bl) * size_tile) + "\t" + str(x_bl * size_tile) + "\t" + str(y_bl * size_tile) +
                               mat_uf)

    with open(L4_Ptrace_path, "w") as L4_Ptrace:
        for idx_cp, cp in enumerate(chiplet_comp):
            L4_Ptrace.write(cp["name"] + "\t")
        for idx_rs, _ in enumerate(space_rects):
            L4_Ptrace.write("Empty_" + str(idx_rs) + "\t")
        L4_Ptrace.write("\n")
        for idx_cp, cp in enumerate(chiplet_comp):
            L4_Ptrace.write(str(cp["power"]) + "\t")
        for idx_rs in enumerate(space_rects):
            L4_Ptrace.write(str(0) + "\t")
        L4_Ptrace.write("\n")

    with open(L5_TIM_path, "w") as L5_TIM:
        L5_TIM.write("# Floorplan for TIM Layer \n")
        L5_TIM.write("TIM\t" + str(W * size_tile) + "\t" + str(H * size_tile) + "\t0.0\t0.0\n")

    with open(LCF_path, "w") as LCF:
        LCF.write("# Layer 0: substrate\n0\nY\nN\n1.06E+06\n3.33\n0.0002\n" + L0_Substrate_path + "\n")
        LCF.write("\n# Layer 1: Epoxy SiO2 underfill with c4 copper pillar\n1\nY\nN\n2.32E+06\n0.625\n0.00007\n" +
                  L1_C4Layer_path + "\n")
        LCF.write("\n# Layer 2: silicon interposer\n2\nY\nN\n1.75E+06\n0.01\n0.00011\n" + L2_Interposer_path + "\n")
        LCF.write("\n# Layer 3: Underfill with ubump\n3\nY\nN\n2.32E+06\n0.625\n1.00E-05\n" + L3_UbumpLayer_path + "\n")
        LCF.write("\n# Layer 4: Chip layer\n4\nY\nY\n1.75E+06\n0.01\n0.00015\n" + L4_ChipLayer_path + "\n")
        LCF.write("\n# Layer 5: TIM\n5\nY\nN\n4.00E+06\n0.25\n2.00E-05\n" + L5_TIM_path + "\n")

    with open(Config_in_path, "r") as Config_in, open(Config_out_path, "w") as Config_out:
        size_spreader = 2 * max(W, H) * size_tile
        size_heatsink = 2 * size_spreader
        r_convec = 0.1 * 0.06 * 0.06 / size_heatsink / size_heatsink  # 0.1 * 0.06 * 0.06 are from default hotspot.config template file
        cfg = Config_in.read()
        cfg = cfg.replace("*size_sink*", str(size_heatsink))
        cfg = cfg.replace("*size_spreader*", str(size_spreader))
        cfg = cfg.replace("*r_convec*", str(r_convec))
        cfg = cfg.replace("*size_grid*", str(size_grid))
        Config_out.write(cfg)

    cmd_list = [
        path.join(dir_hotspot,
                  "hotspot"), "-c", Config_out_path, "-f", L4_ChipLayer_path, "-p", L4_Ptrace_path, "-steady_file", Steady_path,
        "-grid_steady_file", GridSteady_path, "-model_type", "grid", "-detailed_3D", "on", "-grid_layer_file", LCF_path
    ]
    run_res = subprocess.run(cmd_list, capture_output=True, text=True)
    if run_res.returncode != 0:
        raise ValueError("ERROR: hotspot exits error with: {}".format(run_res.stderr))

    # parse grid_steady file
    with open(GridSteady_path, "r") as f:
        grid_steady_file = [line for line in f.readlines() if line.strip()]
    grid_steady = np.ones(shape=(size_grid, size_grid)) * (-1)
    gs_cnt = 0
    for gs in grid_steady_file:
        grid_steady[size_grid - 1 - gs_cnt // size_grid][gs_cnt % size_grid] = float(
            gs.split()[1]) - 273.15  # first value is upper-left; Kelvin to Celsius
        gs_cnt += 1

    assert (grid_steady >= 0).all()

    if clean:
        shutil.rmtree(dir_tmp)

    return grid_steady