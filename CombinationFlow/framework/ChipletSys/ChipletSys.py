import os
import math
import tempfile
import networkx as nx
import numpy as np
import itertools as itl

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

from .HotSpot import get_thermal_hotspot
from .utils import is_rects_overlap
from ..NoC import gen_task_tgff
from .Chiplet import Chiplet


def gen_csys(dir_tgff, bw, cfg_tgff=None, filter=None, clean=False, cfg_csys=None):
    """
        randomly generate ChipletSys instances
        cfg_tgff: "tg_cnt" decides the maximum number of generated instances
        cfg_csys: configuration dict
    """
    # configurations of chiplets
    W_intp = cfg_csys["W_intp"] if "W_intp" in cfg_csys else 50  # width of interposer, unit is tile
    H_intp = cfg_csys["H_intp"] if "H_intp" in cfg_csys else 50  # height of interposer
    max_size_cpl = cfg_csys["max_size_cpl"] if "max_size_cpl" in cfg_csys else 20  # max width/height of single chiplet
    max_wh_ratio_cpl = 2  # the maximum ratio of width to height of chiplet, assume w >= h, so max_wh_ratio_cpl >= 1
    max_area_sys = cfg_csys[
        "max_area_sys"] if "max_area_sys" in cfg_csys else 0.6 * W_intp * H_intp  # max of total area of chiplets
    max_num_cpl = cfg_csys["max_num_cpl"] if "max_num_cpl" in cfg_csys else 30  # max number of chiplets in system
    min_num_cpl = cfg_csys["min_num_cpl"] if "min_num_cpl" in cfg_csys else 5  # min num of chiplets in system
    max_pd_cpl = cfg_csys["max_pd_cpl"] if "max_pd_cpl" in cfg_csys else 0.8  # max power density of one single chiplet
    min_pd_cpl = cfg_csys["min_pd_cpl"] if "min_pd_cpl" in cfg_csys else 0.2  # min power density of one single chiplet
    max_pd_sys = cfg_csys["max_pd_sys"] if "max_pd_sys" in cfg_csys else 0.5  # max power density of whole system
    min_pd_sys = cfg_csys["min_pd_sys"] if "min_pd_sys" in cfg_csys else 0.4  # min power density of whole system
    max_incr_size = cfg_csys[
        "max_incr_size"] if "max_incr_size" in cfg_csys else 30  # max number of try to increase size of chiplets

    csystems = []
    gs = gen_task_tgff(dir_tgff=dir_tgff, bw=bw, cfg=cfg_tgff, filter=filter, clean=clean)
    for g in gs:  # task graph
        cpls = []  # cores in same group are unconnected: [[core 0, core 1, ...], ..., [core x, core y, ...]]
        for c in g:
            flg_connected = False
            if c == 0:
                cpl_this = [0]
            else:
                for c_t in cpl_this:
                    if g.has_edge(c, c_t) or g.has_edge(c_t, c):
                        cpls.append(cpl_this)
                        cpl_this = [c]
                        flg_connected = True
                        break
                if not flg_connected:
                    cpl_this.append(c)
        cpls.append(cpl_this)

        target_num_cpl = np.random.randint(low=min_num_cpl, high=max_num_cpl + 1)
        while True:
            flg_fail = True  # failed to increase/decrease node group
            num_cpl = len(cpls)
            if num_cpl > target_num_cpl:  # merge two unconnected groups
                i_j_cpl = [(i, j) for i in range(num_cpl) for j in range(i + 1, num_cpl)]
                np.random.shuffle(i_j_cpl)
                for i_cpl, j_cpl in i_j_cpl:
                    flg_cnted = False  # connected flag
                    for i_c, j_c in itl.product(cpls[i_cpl], cpls[j_cpl]):
                        if g.has_edge(i_c, j_c) or g.has_edge(j_c, i_c):
                            flg_cnted = True
                            break
                    if not flg_cnted:
                        cpl_ = [cpls[idx_cpl] for idx_cpl in range(num_cpl) if idx_cpl != i_cpl and idx_cpl != j_cpl]
                        cpls = cpl_ + [cpls[i_cpl] + cpls[j_cpl]]
                        flg_fail = False
                        break
                if flg_fail:
                    break

            elif num_cpl < target_num_cpl:
                idx_cpls = [i for i in range(num_cpl) if len(cpls[i]) > 1]
                if not idx_cpls:
                    break
                flg_fail = False
                idx_cpl = np.random.choice(idx_cpls)  # the node group to be split
                cpl_split = cpls[idx_cpl]
                np.random.shuffle(cpl_split)
                cpls = cpls[:idx_cpl] + cpls[idx_cpl + 1:] + [cpl_split[0::2], cpl_split[1::2]]
            else:
                flg_fail = False
                break

        if flg_fail:
            continue
        else:
            assert sum([len(cpl) for cpl in cpls]) == len(g)
            assert set([c for cpl in cpls for c in cpl]) == set(range(len(g)))

        # generate size of chiplet
        size_cpls = []
        for cpl in cpls:
            pin_num = len(cpl)
            size_cpls.append((math.ceil((pin_num + 4) / 4), ) * 2)  # (w, h) tuple, assume a square, 4n - 4 = len(cpl)

        if sum(map(lambda wh: wh[0] * wh[1], size_cpls)) > max_area_sys:  # even smallest chiplets oversize
            break

        for _ in range(max_incr_size):
            idx_cpl = np.random.randint(low=0, high=len(cpls))  # the chiplet to increase size
            h_new = np.random.randint(low=size_cpls[idx_cpl][1],
                                      high=min(size_cpls[idx_cpl][1] + 3, max_size_cpl) + 1)  # control step of increase
            w_new = np.random.randint(low=h_new, high=min(h_new * max_wh_ratio_cpl, max_size_cpl) + 1)  # w >= h for chiplet
            size_cpls_new = size_cpls[:idx_cpl] + [(w_new, h_new)] + size_cpls[idx_cpl + 1:]
            if sum(map(lambda wh: wh[0] * wh[1], size_cpls_new)) <= max_area_sys:
                size_cpls = size_cpls_new

        # random assigned peripheral position
        pin_pos = []
        for idx_cpl, cpl in enumerate(cpls):
            w, h = size_cpls[idx_cpl]
            pin_pos_this = list(
                itl.chain(itl.product([0], range(h - 1)), itl.product(range(w - 1), [h - 1]), itl.product([w - 1], range(1, h)),
                          itl.product(range(1, w), [0])))
            np.random.shuffle(pin_pos_this)
            assert len(pin_pos_this) >= len(cpl)
            pin_pos.append(pin_pos_this[:len(cpl)])

        # generate power
        power_cpls = []
        area_cpl = list(map(lambda wh: wh[0] * wh[1], size_cpls))  # map return a iterator, can not be traversed multiple times
        min_power_sys = int(min_pd_sys * W_intp * H_intp)
        max_power_sys = int(max_pd_sys * W_intp * H_intp)
        min_power_cpls = list(map(lambda area: math.ceil(area * min_pd_cpl), area_cpl))
        max_power_cpls = list(map(lambda area: math.floor(area * max_pd_cpl), area_cpl))

        if sum(min_power_cpls) > max_power_sys or sum(max_power_cpls) < min_power_sys:
            continue
        flg_fail = False
        for idx_cpl in range(len(cpls)):
            max_power_this = min(max_power_cpls[idx_cpl], max_power_sys - sum(min_power_cpls[idx_cpl + 1:]))
            min_power_this = max(min_power_cpls[idx_cpl], min_power_sys - sum(max_power_cpls[idx_cpl + 1:]))
            if min_power_this > max_power_this:
                flg_fail = True
                break
            else:
                power_this = np.random.randint(low=min_power_this, high=max_power_this + 1)
                power_cpls.append(power_this)
                max_power_sys -= power_this
                min_power_sys -= power_this

        if flg_fail:
            continue
        else:
            assert min_pd_sys * W_intp * H_intp <= sum(power_cpls) <= max_pd_sys * W_intp * H_intp
            assert all([
                min_pd_cpl * area_cpl[idx_cpl] <= power_cpls[idx_cpl] <= max_pd_cpl * area_cpl[idx_cpl]
                for idx_cpl in range(len(cpls))
            ])

            # generate chiplets
            assert len(size_cpls) == len(power_cpls) == len(pin_pos) == len(cpls)
            chiplets = [
                Chiplet(name=str(idx_cpl),
                        w=size_cpls[idx_cpl][0],
                        h=size_cpls[idx_cpl][1],
                        power=power_cpls[idx_cpl],
                        pins=pin_pos[idx_cpl]) for idx_cpl in range(len(cpls))
            ]

            pin_map = {}  # mapping from task graph to (idx_cpl, idx_pin_cpl)
            for idx_cpl in range(len(cpls)):  # add nodes/pins
                for idx_pin_cpl in range(len(cpls[idx_cpl])):
                    pin_map[cpls[idx_cpl][idx_pin_cpl]] = (idx_cpl, idx_pin_cpl)
            for (u, v, _) in g.edges(data=True):
                assert pin_map[u][0] != pin_map[v][0]  # pins having edges must not in same chiplet

            csystems.append(ChipletSys(W=W_intp, H=H_intp, chiplets=chiplets, task_graph=g, pin_map=pin_map))
    return csystems


class ChipletSys:
    """
        system composed of chiplets and interposer.
        coordinate axis:
            y
            ^
            |
            |------W-----|
            |            |
            H            |
            |            |
          (0,0)----------| -> x
    """
    def __init__(self, W, H, chiplets, task_graph = None, pin_map = None):
        """
            W, H: widht/height of interposer, unit is tile
            chiplets: a list of chiplet obj
            task_grpah: networkx directed graph with node 0, 1, ... and edge attribute "comm"
            pin_map: map NI idx(key) to (chiplet idx, pin idx of chiplet)(value)
        """
        self.W = W
        self.H = H
        self.chiplets = chiplets
        self.task_graph = task_graph
        self.pin_map = pin_map
        self.check_param()

    def check_param(self):
        if len(self.chiplets) == 0:
            raise ValueError("ERROR: empty chiplets")

        if len(set([c.name for c in self.chiplets])) != len(self.chiplets):  # names for HotSpot
            raise ValueError("ERROR: duplicated names of chiplet")

        if not (type(self.W) == int and type(self.H) == int):
            raise TypeError("ERROR: Interposer width({}) or height({}) is not int".format(type(self.W), type(self.H)))
        if not (self.W > 0 and self.H > 0):
            raise ValueError("ERROR: Non-positive interposer width({}) or height({})".format(self.W, self.H))

        if len(self.chiplets) == 0:
            raise ValueError("ERROR: empty chiplets")
        if not (all((c.w_orig <= self.W and c.h_orig <= self.H) or (c.w_orig <= self.H and c.h_orig <= self.W)
                    for c in self.chiplets)):  # rotation also considered
            raise ValueError("ERROR: Chiplet width or height exceeds")

    @property
    def chiplets_area(self):
        return sum([c.w_orig * c.h_orig for c in self.chiplets])

    def check_placement(self, placement):
        """
            placement format: list of triad: (cx, cy, angle), same order as in chiplet list; cx, cy: bottom left corner of chiplet
        """
        if not len(placement) == len(self.chiplets):
            raise ValueError("ERROR: chiplet number {} in placement not correct".format(len(placement)))

        if not all([(cx >= 0 and cy >= 0 and cx + self.chiplets[idx_c].w(angle) <= self.W
                     and cy + self.chiplets[idx_c].h(angle) <= self.H) for idx_c, (cx, cy, angle) in enumerate(placement)]):
            raise ValueError("ERROR: chiplet(s) exceed interposer boarder", placement)

        if is_rects_overlap([(cx, cy, cx + self.chiplets[idx_c].w(angle), cy + self.chiplets[idx_c].h(angle))
                             for idx_c, (cx, cy, angle) in enumerate(placement)]):  # overlap checking
            self.show_placement(placement=placement)
            raise ValueError("ERROR: the input placement overlapped")

    def show_placement(self, placement, show_grid=True):
        """
            show the placement of chiplets according to chiplets and placement
        """
        plt.imshow(np.ones((self.H, self.W, 3)), cmap="gray")  # white background

        if show_grid:
            for i in range(0, self.H + 1):
                plt.plot((0, self.W), (i, i), alpha=0.5, lw=0.3, color="gray")
            for i in range(0, self.W + 1):
                plt.plot((i, i), (0, self.H), alpha=0.5, lw=0.3, color="gray")

        ax = plt.gca()
        ax.invert_yaxis()

        assert len(placement) == len(self.chiplets)

        cmap = cm.get_cmap("Paired", len(placement))
        colors = cmap(np.linspace(0, 1, len(placement)))

        for idx_p, (cx, cy, angle) in enumerate(placement):
            rect = Rectangle((cx, cy), self.chiplets[idx_p].w(angle), self.chiplets[idx_p].h(angle), color=colors[idx_p])
            ax.add_patch(rect)
            pins = self.chiplets[idx_p].pins(angle)
            for x_p_rel, y_p_rel in pins:
                x_p = cx + x_p_rel
                y_p = cy + y_p_rel
                rect_p = Rectangle((x_p + 0.25, y_p + 0.25), 0.5, 0.5, color="k")
                ax.add_patch(rect_p)

            plt.text(cx, cy, self.chiplets[idx_p].name)

        plt.show()

    @staticmethod
    def show_rect(W, H, rects, show_grid=True):
        """
           rects: list of coordinates, (x_bl, y_bl, x_ur, y_ur)
        """
        plt.imshow(np.ones((H, W, 3)), cmap="gray")  # white background

        if show_grid:
            for i in range(0, H + 1):
                plt.plot((0, W), (i, i), alpha=0.5, lw=0.3, color="gray")
            for i in range(0, W + 1):
                plt.plot((i, i), (0, H), alpha=0.5, lw=0.3, color="gray")

        ax = plt.gca()
        ax.invert_yaxis()

        cmap = cm.get_cmap("Paired", len(rects))
        colors = cmap(np.linspace(0, 1, len(rects)))
        # np.random.shuffle(colors)

        for idx_r, (x_bl, y_bl, x_ur, y_ur) in enumerate(rects):
            rect = Rectangle((x_bl, y_bl), x_ur - x_bl, y_ur - y_bl, alpha=1, color=colors[idx_r])
            ax.add_patch(rect)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

    def get_xy_pin(self, pin, placement):
        """
            Get id of chiplet + id of pin by using "p", then get xy with angle

            pin: idx in task graph
        """
        idx_cpl, idx_pin_cpl = self.pin_map[pin]
        x_cpl, y_cpl, angle = placement[idx_cpl]
        xy_rel = self.chiplets[idx_cpl].pins(angle)[idx_pin_cpl]  # relative to chiplet
        return (x_cpl + xy_rel[0], y_cpl + xy_rel[1])

    def eval_thermal(self, dir_hotspot, tile_size, placement, visualize=False, cfg=None):
        """
            dir_hotspot: ABSOLUTE path of executable file HotSpot should be: exec_dir/hotspot
            tile_size: the size of one tile, unit: meter
        """
        if cfg is None:
            cfg = {}
        if "W_tile" not in cfg:
            cfg["W_tile"] = self.W
        if "H_tile" not in cfg:
            cfg["H_tile"] = self.H
        assert "tile_size" not in cfg
        cfg["tile_size"] = tile_size
        assert type(self.W) == int

        chiplet_comp = []

        for idx_p, (cx, cy, angle) in enumerate(placement):
            chiplet_comp.append({
                "width": self.chiplets[idx_p].w(angle),
                "height": self.chiplets[idx_p].h(angle),
                "left_x": cx,
                "bottom_y": cy,
                "power": self.chiplets[idx_p].power,
                "name": self.chiplets[idx_p].name
            })
        grid_steady = get_thermal_hotspot(chiplet_comp=chiplet_comp, dir_hotspot=dir_hotspot, clean=True, config=cfg)
        if visualize:
            cmap = LinearSegmentedColormap.from_list('byr', ["b", "y", "r"], N=20)
            im = plt.imshow(grid_steady, cmap=cmap)
            plt.colorbar(im)
            ax = plt.gca()
            ax.invert_yaxis()
            plt.show()

        return grid_steady  # return the whole temperature matrix

    def show_graph(self, show_pin):
        """
            if show_pin is True, then node name showed is (chiplet id, pin id), else is core id
        """
        if show_pin:
            g = nx.relabel_nodes(self.task_graph, self.pin_map, copy=True)
        else:
            g = self.task_graph

        fd, img_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        PG = nx.nx_pydot.to_pydot(g)
        for e in PG.get_edges():
            e.set_label("{}".format(e.get_attributes()["comm"]))
        PG.write_png(img_path)
        img = plt.imread(img_path)
        plt.axis('off')
        plt.imshow(img)
        plt.show()
        os.remove(img_path)