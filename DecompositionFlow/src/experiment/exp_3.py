from typing import List

from .. import dataset
from ..model import Block, eval_ev_ppc, get_bd_bid, get_cost_detail

core = Block(name="core", area=5.05, node=7)
L3 = Block(name="L3", area=16.8, node=7)  # 16 MB L3
ddr = Block(name="ddr", area=33.74, node=14)  # dual channel
pcie = Block(name="pcie", area=35.26, node=14)  # x16 64 GB/s

if __name__ == "__main__":
    bdg_all = dataset.AMD()
    _, bid_all = get_bd_bid(bdg_all=bdg_all)
    bc_amd = [[{ddr, pcie}, {core, L3}], [{ddr, pcie}, {core, L3}], [{ddr, pcie}, {core, L3}], [{ddr, pcie}, {core, L3}],
              [{ddr, pcie}, {core, L3}], [{ddr, pcie}, {core, L3}]]
    cn_amd = [[1, 2], [1, 4], [1, 6], [1, 8], [1, 1], [1, 2]]

    # AMD vs. Chipletizer: 500k
    vol_single = 500 * 1000
    bc_new_500k = [[{L3, core}, {pcie, ddr}], [{L3, core}, {pcie, ddr}], [{L3, core}, {pcie, ddr}], [{L3, core}, {pcie, ddr}],
                   [{L3, core}, {pcie, ddr}], [{L3, core}, {pcie, ddr}]]
    cn_new_500k = [[2, 4], [4, 4], [6, 4], [8, 4], [1, 1], [2, 1]]
    print(
        "Official 500k: ",
        eval_ev_ppc(bdg_all=bdg_all,
                    vol_all=[vol_single] * len(bdg_all),
                    w_power=0,
                    w_perf=0,
                    w_cost=1,
                    type_pkg="SI",
                    bc_all=bc_amd,
                    cn_all=cn_amd,
                    bid_all=bid_all))
    print(
        "My 500k: ",
        eval_ev_ppc(bdg_all=bdg_all,
                    vol_all=[vol_single] * len(bdg_all),
                    w_power=0,
                    w_perf=0,
                    w_cost=1,
                    type_pkg="SI",
                    bc_all=bc_new_500k,
                    cn_all=cn_new_500k,
                    bid_all=bid_all))

    # AMD vs. Chipletizer: 10Ｍ
    vol_single = 10 * 1000 * 1000
    bc_new_10M = [[{ddr, pcie}, {core, L3}], [{ddr, pcie}, {core, L3}], [{ddr, pcie}, {core, L3}], [{ddr, pcie}, {core, L3}],
                  [{ddr, pcie}, {core, L3}], [{ddr, pcie}, {core, L3}]]
    cn_new_10M = [[2, 1], [2, 2], [2, 3], [2, 4], [1, 1], [1, 1]]
    print(
        "Official 10M: ",
        eval_ev_ppc(bdg_all=bdg_all,
                    vol_all=[vol_single] * len(bdg_all),
                    w_power=0,
                    w_perf=0,
                    w_cost=1,
                    type_pkg="SI",
                    bc_all=bc_amd,
                    cn_all=cn_amd,
                    bid_all=bid_all))

    print(
        "My 10M: ",
        eval_ev_ppc(bdg_all=bdg_all,
                    vol_all=[vol_single] * len(bdg_all),
                    w_power=0,
                    w_perf=0,
                    w_cost=1,
                    type_pkg="SI",
                    bc_all=bc_new_10M,
                    cn_all=cn_new_10M,
                    bid_all=bid_all))