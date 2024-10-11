import pickle

import numpy as np
import pandas as pd


def cost():
    for q in [500 * 1000, 10 * 1000 * 1000]:
        with open("/research/Chipletization/log/exp_1/{}/res.pickle".format(q), "rb") as f:
            ppc, cd = pickle.load(f)
        data = []
        for idx_comp, comp in enumerate(["AMD", "HiSilicon", "Intel", "Nvidia", "Rockchip"]):
            """
                    company    strategy    NRE Cost of Packages    NRE cost of SoC Chips/Chiplets    RE Cost of Raw Packages ...
                0
                1
            """
            norm = cd["CP"][idx_comp].sum().sum()  # use chopin to normalize
            for st in ["CP", "M", "RF", "BP", "FG", "C"]:
                d = (cd[st][idx_comp].sum() / norm).to_dict()
                d["strategy"] = st
                d["company"] = comp
                data.append(d)
        df = pd.DataFrame(data=data)
        df.to_csv("results/{}.csv".format(q))


def cost_raw():
    """
        obtain the raw cost
    """
    for q in [500 * 1000, 10 * 1000 * 1000]:
        with open("/research/Chipletization/log/exp_1/{}/res.pickle".format(q), "rb") as f:
            ppc, cd = pickle.load(f)
        data = []
        for idx_comp, comp in enumerate(["AMD", "HiSilicon", "Intel", "Nvidia", "Rockchip"]):
            """
                    company    strategy    NRE Cost of Packages    NRE cost of SoC Chips/Chiplets    RE Cost of Raw Packages ...
                0
                1
            """
            for st in ["CP", "M", "RF", "BP", "FG", "C"]:
                num_product = len(cd[st][idx_comp])
                d = {"cost": cd[st][idx_comp].sum().sum() / num_product}
                d["strategy"] = st
                d["company"] = comp
                data.append(d)
        df = pd.DataFrame(data=data)
        df.to_csv("{}_raw.csv".format(q))


def cost_ratio():
    for q in [500 * 1000, 10 * 1000 * 1000]:
        with open("/research/Chipletization/log/exp_1/{}/res.pickle".format(q), "rb") as f:
            ppc, cd = pickle.load(f)

        for st in ["CP", "M", "RF", "BP", "FG", "C"]:
            r = []
            for idx_comp, comp in enumerate(["AMD", "HiSilicon", "Intel", "Nvidia", "Rockchip"]):
                norm = cd["CP"][idx_comp].sum().sum()  # use chopin to normalize
                r.append(1 - norm / cd[st][idx_comp].sum().sum())  # cost reudction
            print(st, np.average(r) * 100)


def power():
    for q in [500 * 1000, 10 * 1000 * 1000]:
        with open("/research/Chipletization/log/exp_1/{}/res.pickle".format(q), "rb") as f:
            ppc, _ = pickle.load(f)
        for st in ["CP", "M", "RF", "BP", "FG", "C"]:
            power_st = []
            for idx_comp, comp in enumerate(["AMD", "HiSilicon", "Intel", "Nvidia", "Rockchip"]):
                power_st += ppc[st]["power"][idx_comp]
            print(st, np.average(power_st))


if __name__ == "__main__":
    cost()