from itertools import product
import os
import networkx as nx
from typing import List, Set
from multiprocessing import Pool
from .. import utils
from .. import dataset
from ..model import Block, Chiplet, Package,make_package, SI,get_cost
from ..solver import IndiSolver, HoliSolver

dir_log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../log")


def test_IndiSolver():
    params = {}
    params["type_pkg"] = "SI"
    # params of IndiSolver
    params["indi_max_try"] = 10
    params["indi_T_start"] = 1
    params["indi_T_end"] = 0.01
    params["indi_alpha"] = 0.95
    params["w_power"] = 0
    params["w_perf"] = 0
    params["w_cost"] = 1

    sunny_cove = Block(name="sunny_cove", area=9.04, node=10)
    pcie = Block(name="pcie", area=7.82, node=10)  # 16 lanes
    upi = Block(name="upi", area=18.75, node=10)
    ddr = Block(name="ddr", area=27.88, node=10)  # 2 channels
    xeon_platinum_8380 = nx.DiGraph()
    xeon_platinum_8380.add_nodes_from(range(0, 40), block=sunny_cove)
    xeon_platinum_8380.add_nodes_from(range(40, 44), block=pcie)
    xeon_platinum_8380.add_nodes_from(range(44, 47), block=upi)
    xeon_platinum_8380.add_nodes_from(range(47, 51), block=ddr)

    for i in range(40):
        for j in range(40):
            if i != j:
                xeon_platinum_8380.add_edge(i, j, comm=128 / 8 * 2, ener_eff=0, perf_penal=0)

    isolver = IndiSolver(bdg=xeon_platinum_8380, vol=500 * 1000, dir_log=dir_log, params=params)
    isolver.init_opt()
    print(isolver.opt())


@utils.timing
def test_HoliSolver_1():
    params = {}
    params["type_pkg"] = "SI"
    params["holi_max_try"] = 10
    params["holi_num_init_sample"] = 20
    params["holi_T_start"] = 1
    params["holi_T_end"] = 0.01
    params["holi_alpha"] = 0.95
    params["w_power"] = 1
    params["w_perf"] = 1
    params["w_cost"] = 1

    sunny_cove = Block(name="sunny_cove", area=9.04, node=10)
    pcie = Block(name="pcie", area=7.82, node=10)  # 16 lanes
    upi = Block(name="upi", area=18.75, node=10)
    ddr = Block(name="ddr", area=27.88, node=10)  # 2 channels
    xeon_platinum_8380 = nx.DiGraph()
    xeon_platinum_8380.add_nodes_from(range(0, 40), block=sunny_cove)
    xeon_platinum_8380.add_nodes_from(range(40, 44), block=pcie)
    xeon_platinum_8380.add_nodes_from(range(44, 46), block=upi)
    xeon_platinum_8380.add_nodes_from(range(46, 50), block=ddr)
    xeon_platinum_8368 = nx.DiGraph()
    xeon_platinum_8368.add_nodes_from(range(0, 38), block=sunny_cove)
    xeon_platinum_8368.add_nodes_from(range(38, 42), block=pcie)
    xeon_platinum_8368.add_nodes_from(range(42, 46), block=upi)
    xeon_platinum_8368.add_nodes_from(range(46, 50), block=ddr)

    hsolver = HoliSolver(bdg_all=[xeon_platinum_8380, xeon_platinum_8368],
                         vol_all=[500 * 1000, 700 * 1000],
                         bc_all=[[{upi, ddr}, {sunny_cove, pcie}], [{sunny_cove, pcie}, {upi, ddr}]],
                         cn_all=[[2, 4], [1, 4]],
                         ppc_all=[(-1, -2, -3), (-4, -5, -6)],
                         dir_log=dir_log,
                         params=params)
    print(hsolver.get_ev_naive(state_bc=hsolver.state_bc_cur, state_cn=hsolver.state_cn_cur))
    print(
        hsolver.adjust_cn(bc_target={sunny_cove, pcie},
                          state_bc=hsolver.state_bc_cur,
                          state_cn=hsolver.state_cn_cur,
                          ppc=[(-1, -2, -3), (-4, -5, -6)]))
    # hsolver.get_ev_naive()


def test_HoliSolver_2():
    params = {}
    params["type_pkg"] = "SI"
    params["holi_max_try"] = 10
    params["holi_num_init_sample"] = 20
    params["holi_T_start"] = 1
    params["holi_T_end"] = 0.01
    params["holi_alpha"] = 0.95
    params["w_power"] = 1
    params["w_perf"] = 1
    params["w_cost"] = 1

    bdg_all = dataset.Intel()
    vol_all = [500 * 1000] * len(bdg_all)
    sunny_cove = Block(name="sunny_cove", area=9.04, node=10)
    pcie = Block(name="pcie", area=7.82, node=10)  # 16 lanes
    upi = Block(name="upi", area=18.75, node=10)
    ddr = Block(name="ddr", area=27.88, node=10)  # 2 channels

    bc_all = [[{sunny_cove}, {pcie, upi, ddr}] for _ in range(len(bdg_all))]
    cn_all = [[1, 1] for _ in range(len(bdg_all))]
    hsolver = HoliSolver(bdg_all=bdg_all, vol_all=vol_all, bc_all=bc_all, cn_all=cn_all, dir_log=dir_log, params=params)
    hsolver.adjust_cn(bc_target={sunny_cove}, state_bc=hsolver.state_bc_cur, state_cn=hsolver.state_cn_cur, ppc=hsolver.ppc_cur)
    hsolver.opt()

if __name__ == "__main__":
    # test_IndiSolver()
    test_HoliSolver_2()
