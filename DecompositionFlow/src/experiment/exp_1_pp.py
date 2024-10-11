import pickle

import numpy as np
import pandas as pd

with open("/research/Chipletization/log/exp_1/{}/res.pickle".format(500*1000), "rb") as f:
    ppc_500k, _ = pickle.load(f)

for q in [500 * 1000, 10 * 1000 * 1000]:
        with open("/research/Chipletization/log/exp_1/{}/res.pickle".format(q), "rb") as f:  # exp_1_pp?
            ppc, _ = pickle.load(f)    
        data = []
        for idx_comp, comp in enumerate(["AMD", "HiSilicon", "Intel", "Nvidia", "Rockchip"]):
            """
                    company    strategy    NRE Cost of Packages    NRE cost of SoC Chips/Chiplets    RE Cost of Raw Packages ...
                0
                1
            """
            for st in ["CP", "M", "RF", "BP", "FG", "C"]:
                power_st_norm = np.average(ppc_500k["FG"]["power"][idx_comp])
                perf_st_norm = np.average(ppc_500k["FG"]["perf"][idx_comp])

                power_st_comp = ppc[st]["power"][idx_comp]
                perf_st_comp = ppc[st]["perf"][idx_comp]
                assert len(power_st_comp) == len(perf_st_comp)
                
                d = {"power": np.average(power_st_comp) / power_st_norm, "perf": np.average(perf_st_comp) / perf_st_norm}
                d["strategy"] = st
                d["company"] = comp
                data.append(d)
        df = pd.DataFrame(data=data)
        df.to_csv("results/{}_pp.csv".format(q))