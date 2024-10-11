import pickle

import pandas as pd

for q in [500 * 1000, 10 * 1000 * 1000]:
    with open("/research/Chipletization/log/exp_1/{}/res.pickle".format(q), "rb") as f:
        _, cd_i_h = pickle.load(f)
    with open("/research/Chipletization/log/exp_2/0/{}/res.pickle".format(q), "rb") as f:
        _, cd_h = pickle.load(f)
    with open("/research/Chipletization/log/exp_2/1/{}/res.pickle".format(q), "rb") as f:
        _, cd_i = pickle.load(f)

    for idx_comp, comp in enumerate(["AMD", "HiSilicon", "Intel", "Nvidia", "Rockchip"]):
        """
                company    strategy    NRE Cost of Packages    NRE cost of SoC Chips/Chiplets    RE Cost of Raw Packages ...
            0
            1
        """
        norm = cd_i_h["CP"][idx_comp].sum().sum()  # use chopin to normalize
        print(comp, "indi: ", cd_i["CP"][idx_comp].sum().sum() / norm)
        print(comp, "holi: ", cd_h["CP"][idx_comp].sum().sum() / norm)