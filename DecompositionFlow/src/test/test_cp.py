from ..model import Block, eval_ev_ppc
import networkx as nx
import os
from typing import List, Dict, Tuple, Set
from .. import dataset
from ..solver import chiplet_partition

core = Block(name="core", area=20, node=7)
L3 = Block(name="L3", area=4, node=7)
io = Block(name="io", area=104, node=14)

epyc_0 = nx.DiGraph()
epyc_0.add_nodes_from(range(0, 16), block=core)
epyc_0.add_nodes_from(range(16, 20), block=L3)
epyc_0.add_nodes_from(range(20, 24), block=io)

epyc_1 = nx.DiGraph()
epyc_1.add_nodes_from(range(0, 32), block=core)
epyc_1.add_nodes_from(range(32, 40), block=L3)
epyc_1.add_nodes_from(range(40, 44), block=io)

epyc_2 = nx.DiGraph()
epyc_2.add_nodes_from(range(0, 48), block=core)
epyc_2.add_nodes_from(range(48, 60), block=L3)
epyc_2.add_nodes_from(range(60, 64), block=io)

epyc_3 = nx.DiGraph()
epyc_3.add_nodes_from(range(0, 64), block=core)
epyc_3.add_nodes_from(range(64, 80), block=L3)
epyc_3.add_nodes_from(range(80, 84), block=io)

bdg_all = [epyc_0, epyc_1, epyc_2, epyc_3]


def get_bd_bid(bdg_all: List[nx.DiGraph]):
    bd_all: List[Dict[Block, int]] = []  # each item is block description of the SoC: {"cpu":2, "gpu":4}
    bid_all: List[Dict[Block, List[int]]] = []
    for bdg in bdg_all:
        bd = {}
        bid = {}
        for nid, attr in bdg.nodes(data=True):
            blk = attr["block"]
            if blk not in bd:
                bd[blk] = 0
                bid[blk] = []
            bd[blk] += 1
            bid[blk].append(nid)
        bd_all.append(bd)
        bid_all.append(bid)
    return bd_all, bid_all


def test_1():
    params = {}
    # params of chiplet_partition
    params["num_cpu"] = 32
    params["indi_pnum"] = 2
    params["dir_log_root"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../log", "test")
    import shutil
    shutil.rmtree(params["dir_log_root"])
    os.mkdir(params["dir_log_root"])
    params["type_pkg"] = "SI"
    # params of IndiSolver
    params["indi_max_try"] = 10
    params["indi_T_start"] = 1
    params["indi_T_end"] = 0.01
    params["indi_alpha"] = 0.95
    params["w_power"] = 0
    params["w_perf"] = 0
    params["w_cost"] = 1
    # params of HoliSolver
    params["holi_pnum"] = 8
    params["holi_num_init_sample"] = 20
    params["holi_max_try"] = 10
    params["holi_T_start"] = 1
    params["holi_T_end"] = 0.01
    params["holi_alpha"] = 0.95

    bdg_all = dataset.HiSilicon()
    # bdg_all = dataset.Newform()
    vol_all = [20 * 1000 * 1000] * len(bdg_all)
    chiplet_partition(bdg_all=bdg_all, vol_all=vol_all, params=params)


if __name__ == "__main__":
    test_1()