import pickle
import numpy as np

num_product = [6, 6, 10, 10, 6]  # AMD, HiSilicon, Intel, Rockchip, Nvidia


def CI_BP_pl():
    power_ratio = []
    lat_ratio = []
    for q in [500 * 1000, 10 * 1000 * 1000]:
        power_CI = []
        power_BP = []
        lat_CI = []
        lat_BP = []
        with open("/research/Chipletization/log/exp_1_pp/{}/res.pickle".format(q), "rb") as f:
            ppc, _ = pickle.load(f)
        for idx_comp, comp in enumerate(["AMD", "HiSilicon", "Intel", "Nvidia", "Rockchip"]):
            power_CI.extend(ppc["CP"]["power"][idx_comp])
            power_BP.extend(ppc["BP"]["power"][idx_comp])
            lat_CI.extend(ppc["CP"]["perf"][idx_comp])
            lat_BP.extend(ppc["BP"]["perf"][idx_comp])
            print(q, comp, sum(ppc["CP"]["power"][idx_comp]) >= sum(ppc["BP"]["power"][idx_comp]))
            print(q, comp, sum(ppc["CP"]["perf"][idx_comp]) >= sum(ppc["BP"]["perf"][idx_comp]))

        power_ratio.append(np.average(power_BP) / np.average(power_CI))
        lat_ratio.append(np.average(lat_BP) / np.average(lat_CI))
    print("%.2f" % (np.average(power_ratio)))
    print("%.2f" % (np.average(lat_ratio)))


def avg_pl():
    power_CI = []
    power_M = []
    power_RF = []
    power_BP = []
    power_FG = []
    power_CO = []

    lat_CI = []
    lat_M = []
    lat_RF = []
    lat_BP = []
    lat_FG = []
    lat_CO = []

    for q in [500 * 1000, 10 * 1000 * 1000]:
        with open("/research/Chipletization/log/exp_1_pp/{}/res.pickle".format(q), "rb") as f:
            ppc, _ = pickle.load(f)
        for idx_comp, comp in enumerate(["AMD", "HiSilicon", "Intel", "Nvidia", "Rockchip"]):
            power_CI.extend(ppc["CP"]["power"][idx_comp])
            power_M.extend(ppc["M"]["power"][idx_comp])
            power_RF.extend(ppc["RF"]["power"][idx_comp])
            power_BP.extend(ppc["BP"]["power"][idx_comp])
            power_FG.extend(ppc["FG"]["power"][idx_comp])
            power_CO.extend(ppc["C"]["power"][idx_comp])

            lat_CI.extend(ppc["CP"]["perf"][idx_comp])
            lat_M.extend(ppc["M"]["perf"][idx_comp])
            lat_RF.extend(ppc["RF"]["perf"][idx_comp])
            lat_BP.extend(ppc["BP"]["perf"][idx_comp])
            lat_FG.extend(ppc["FG"]["perf"][idx_comp])
            lat_CO.extend(ppc["C"]["perf"][idx_comp])

        power_CI_avg = np.average(power_CI)
        power_M_avg = np.average(power_M)
        power_RF_avg = np.average(power_RF)
        power_BP_avg = np.average(power_BP)
        power_FG_avg = np.average(power_FG)
        power_CO_avg = np.average(power_CO)

        lat_CI_avg = np.average(lat_CI)
        lat_M_avg = np.average(lat_M)
        lat_RF_avg = np.average(lat_RF)
        lat_BP_avg = np.average(lat_BP)
        lat_FG_avg = np.average(lat_FG)
        lat_CO_avg = np.average(lat_CO)

        print(q, "M", "power %.2f" % (power_CI_avg / power_M_avg), " lat %.2f" % (lat_CI_avg / lat_M_avg))
        print(q, "RF", "power %.2f" % (power_CI_avg / power_RF_avg), " lat %.2f" % (lat_CI_avg / lat_RF_avg))
        print(q, "BP", "power %.2f" % (power_CI_avg / power_BP_avg), " lat %.2f" % (lat_CI_avg / lat_BP_avg))
        print(q, "FG", "power %.2f" % (power_CI_avg / power_FG_avg), " lat %.2f" % (lat_CI_avg / lat_FG_avg))
        print(q, "CO", "power %.2f" % (power_CI_avg / power_CO_avg), " lat %.2f" % (lat_CI_avg / lat_CO_avg))


def avg_cost_500k_1():
    CI, M, RF, BP, FG, CO = [], [], [], [], [], []
    for q in [500 * 1000]:
        with open("/research/Chipletization/log/exp_1/{}/res.pickle".format(q), "rb") as f:
            ppc, _ = pickle.load(f)
        for idx_comp, comp in enumerate(["AMD", "HiSilicon", "Intel", "Nvidia", "Rockchip"]):
            CI.extend(ppc["CP"]["cost"][idx_comp])
            M.extend(ppc["M"]["cost"][idx_comp])
            RF.extend(ppc["RF"]["cost"][idx_comp])
            BP.extend(ppc["BP"]["cost"][idx_comp])
            FG.extend(ppc["FG"]["cost"][idx_comp])
            CO.extend(ppc["C"]["cost"][idx_comp])
    print("M", "%.2f" % (1 - np.average(CI) / np.average(M)))
    print("RF", "%.2f" % (1 - np.average(CI) / np.average(RF)))
    print("BP", "%.2f" % (1 - np.average(CI) / np.average(BP)))
    print("FG", "%.2f" % (1 - np.average(CI) / np.average(FG)))
    print("CO", "%.2f" % (1 - np.average(CI) / np.average(CO)))


def avg_cost_10M_1():
    CI, M, RF, BP, FG, CO = [], [], [], [], [], []
    for q in [10 * 1000 * 1000]:
        with open("/research/Chipletization/log/exp_1/{}/res.pickle".format(q), "rb") as f:
            ppc, _ = pickle.load(f)
        for idx_comp, comp in enumerate(["AMD", "HiSilicon", "Intel", "Nvidia", "Rockchip"]):
            CI.extend(ppc["CP"]["cost"][idx_comp])
            M.extend(ppc["M"]["cost"][idx_comp])
            RF.extend(ppc["RF"]["cost"][idx_comp])
            BP.extend(ppc["BP"]["cost"][idx_comp])
            FG.extend(ppc["FG"]["cost"][idx_comp])
            CO.extend(ppc["C"]["cost"][idx_comp])
    print("M", "%.2f" % (1 - np.average(CI) / np.average(M)))
    print("RF", "%.2f" % (1 - np.average(CI) / np.average(RF)))
    print("BP", "%.2f" % (1 - np.average(CI) / np.average(BP)))
    print("FG", "%.2f" % (1 - np.average(CI) / np.average(FG)))
    print("CO", "%.2f" % (1 - np.average(CI) / np.average(CO)))


def avg_cost_500k_2():
    for q in [500 * 1000]:
        M, RF, BP, FG, CO = [], [], [], [], []
        with open("/research/Chipletization/log/exp_1/{}/res.pickle".format(q), "rb") as f:
            ppc, _ = pickle.load(f)
        for idx_comp, comp in enumerate(["AMD", "HiSilicon", "Intel", "Nvidia", "Rockchip"]):
            M_ratio = 1 - np.average(ppc["CP"]["cost"][idx_comp]) / np.average(ppc["M"]["cost"][idx_comp])
            RF_ratio = 1 - np.average(ppc["CP"]["cost"][idx_comp]) / np.average(ppc["RF"]["cost"][idx_comp])
            BP_ratio = 1 - np.average(ppc["CP"]["cost"][idx_comp]) / np.average(ppc["BP"]["cost"][idx_comp])
            FG_ratio = 1 - np.average(ppc["CP"]["cost"][idx_comp]) / np.average(ppc["FG"]["cost"][idx_comp])
            CO_ratio = 1 - np.average(ppc["CP"]["cost"][idx_comp]) / np.average(ppc["C"]["cost"][idx_comp])

            M.append(M_ratio)
            RF.append(RF_ratio)
            BP.append(BP_ratio)
            FG.append(FG_ratio)
            CO.append(CO_ratio)

        print("M", "%.4f" % (np.average(M) * 100))
        print("RF", "%.4f" % (np.average(RF) * 100))
        print("BP", "%.4f" % (np.average(BP) * 100))
        print("FG", "%.4f" % (np.average(FG) * 100))
        print("CO", "%.4f" % (np.average(CO) * 100))


def avg_cost_10M_2():
    for q in [10 * 1000 * 1000]:
        M, RF, BP, FG, CO = [], [], [], [], []
        with open("/research/Chipletization/log/exp_1/{}/res.pickle".format(q), "rb") as f:
            ppc, _ = pickle.load(f)
        for idx_comp, comp in enumerate(["AMD", "HiSilicon", "Intel", "Nvidia", "Rockchip"]):
            M_ratio = 1 - np.average(ppc["CP"]["cost"][idx_comp]) / np.average(ppc["M"]["cost"][idx_comp])
            RF_ratio = 1 - np.average(ppc["CP"]["cost"][idx_comp]) / np.average(ppc["RF"]["cost"][idx_comp])
            BP_ratio = 1 - np.average(ppc["CP"]["cost"][idx_comp]) / np.average(ppc["BP"]["cost"][idx_comp])
            FG_ratio = 1 - np.average(ppc["CP"]["cost"][idx_comp]) / np.average(ppc["FG"]["cost"][idx_comp])
            CO_ratio = 1 - np.average(ppc["CP"]["cost"][idx_comp]) / np.average(ppc["C"]["cost"][idx_comp])

            M.append(M_ratio)
            RF.append(RF_ratio)
            BP.append(BP_ratio)
            FG.append(FG_ratio)
            CO.append(CO_ratio)

        print("M", "%.4f" % (np.average(M) * 100))
        print("RF", "%.4f" % (np.average(RF) * 100))
        print("BP", "%.4f" % (np.average(BP) * 100))
        print("FG", "%.4f" % (np.average(FG) * 100))
        print("CO", "%.4f" % (np.average(CO) * 100))


def NRE_percent():
    with open("/research/Chipletization/log/exp_1/{}/res.pickle".format(500 * 1000), "rb") as f:
        ppc, cd = pickle.load(f)
    k = "NRE cost of SoC Chips/Chiplets"
    M = cd["M"]
    percent_M = []
    for m in M:
        percent_M.append(m[k].sum() / m.sum().sum())
    print(np.average(percent_M))

    BP = cd["BP"]
    percent_BP = []
    for bp in BP:
        percent_BP.append(bp[k].sum() / bp.sum().sum())
    print(np.average(percent_BP))


def reuse_first_comparison():
    with open("/research/Chipletization/log/exp_1/{}/res.pickle".format(500 * 1000), "rb") as f:
        ppc, cd = pickle.load(f)
    k1 = "NRE cost of SoC Chips/Chiplets"
    k2 = "NRE Cost of Packages"
    k3 = "RE Cost of Wasted KGDs"
    k4 = "RE Cost of Package Defects"
    NRE_ratio = []

    RF = cd["RF"]
    M = cd["M"]

    for rf, m in zip(RF, M):
        # NRE_ratio.append((rf[k1].sum() + rf[k2].sum()) / (m[k1].sum() + m[k2].sum()))
        NRE_ratio.append(rf[k1].sum() / m[k1].sum())

    print((1 - np.average(NRE_ratio)) * 100, "%")

    print(RF[4].sum().sum() / M[4].sum().sum() * 100 - 100, "%")

    print(100 - RF[4][k1].sum() / M[4][k1].sum() * 100, "%")

    print(RF[4][k2].sum() / M[4][k2].sum() * 100 - 100, "%")

    wasted_KGD_ratio = []
    for rf, m in zip(RF, M):
        wasted_KGD_ratio.append((rf[k3].sum() + rf[k4].sum()) / (m[k3].sum() + m[k4].sum()))
    print(np.average(wasted_KGD_ratio), "x")


def finest_granularity_comparison():
    with open("/research/Chipletization/log/exp_1/{}/res.pickle".format(500 * 1000), "rb") as f:
        ppc, cd = pickle.load(f)
    k1 = "NRE cost of SoC Chips/Chiplets"

    FG = cd["FG"][4]
    M = cd["M"][4]

    print(FG[k1].sum() / M[k1].sum())


def chopin_comparison():
    with open("/research/Chipletization/log/exp_1/{}/res.pickle".format(500 * 1000), "rb") as f:
        ppc, cd = pickle.load(f)
    k1 = "NRE cost of SoC Chips/Chiplets"
    Chopin = cd["C"]
    M = cd["M"]

    NRE_ratio = []
    for chopin, m in zip(Chopin, M):
        NRE_ratio.append(chopin[k1].sum() / m[k1].sum())
    print((1 - np.average(NRE_ratio)) * 100, "%")


def chipletizer_comparison():
    with open("/research/Chipletization/log/exp_1/{}/res.pickle".format(500 * 1000), "rb") as f:
        ppc, cd = pickle.load(f)
    k1 = "NRE cost of SoC Chips/Chiplets"
    Chipletizer = cd["CP"]
    RF = cd["RF"]
    NRE_ratio = []
    for chipletizer, rf in zip(Chipletizer, RF):
        NRE_ratio.append(chipletizer[k1].sum() / rf[k1].sum())
    print((1 - np.average(NRE_ratio)) * 100, "%")


def monolithic_comparison():
    with open("/research/Chipletization/log/exp_1/{}/res.pickle".format(10 * 1000 * 1000), "rb") as f:
        ppc, cd = pickle.load(f)
    k = "RE Cost of Die Defects"
    m = cd["M"][0]
    print("%.4f" % (np.average(m[k].sum() / m.sum().sum()) * 100))


def balanced_partition_comparison():
    with open("/research/Chipletization/log/exp_1/{}/res.pickle".format(10 * 1000 * 1000), "rb") as f:
        ppc, cd = pickle.load(f)
    k1 = "NRE cost of SoC Chips/Chiplets"
    k2 = "RE Cost of Raw Dies"
    bp = cd["BP"]
    chipletizer = cd["CP"]
    print("%.4f" % (np.average(bp[1][k1].sum() / bp[1].sum().sum()) * 100))
    print("%.4f" % (np.average(bp[0][k2].sum() / chipletizer[0][k2].sum())))


def rf_fg_comparison():
    with open("/research/Chipletization/log/exp_1/{}/res.pickle".format(500 * 1000), "rb") as f:
        _, cd_500k = pickle.load(f)

    with open("/research/Chipletization/log/exp_1/{}/res.pickle".format(10 * 1000 * 1000), "rb") as f:
        _, cd_10M = pickle.load(f)
    k1 = "RE Cost of Package Defects"
    k2 = "RE Cost of Wasted KGDs"
    rf_500k = cd_500k["RF"]
    fg_500k = cd_500k["FG"]
    rf_10M = cd_10M["RF"]
    fg_10M = cd_10M["FG"]

    rf_pf_ratio = []
    for _rf_500k, _rf_10M in zip(rf_500k, rf_10M):
        rf_pf_ratio.append(((_rf_10M[k1].sum() + _rf_10M[k2].sum()) / _rf_10M.sum().sum()) /
                           ((_rf_500k[k1].sum() + _rf_500k[k2].sum()) / _rf_500k.sum().sum()))
    print("%.4f" % (np.average(rf_pf_ratio)))

    fg_pf_ratio = []
    for _fg_500k, _fg_10M in zip(fg_500k, fg_10M):
        fg_pf_ratio.append(((_fg_10M[k1].sum() + _fg_10M[k2].sum()) / _fg_10M.sum().sum()) /
                           ((_fg_500k[k1].sum() + _fg_500k[k2].sum()) / _fg_500k.sum().sum()))
    print("%.4f" % (np.average(fg_pf_ratio)))


def power_comparison():
    def sum_(l):
        return sum([sum(ll) for ll in l])
    
    def max_(l):
        return max([max(ll) for ll in l])

    with open("/research/Chipletization/log/exp_1_pp/{}/res.pickle".format(500 * 1000), "rb") as f:
        ppc_500k, _ = pickle.load(f)

    with open("/research/Chipletization/log/exp_1_pp/{}/res.pickle".format(10 * 1000 * 1000), "rb") as f:
        ppc_10M, _ = pickle.load(f)

    power_500k_chipletizer = sum_(ppc_500k["CP"]["power"])
    power_500k_rf = sum_(ppc_500k["RF"]["power"])
    power_500k_fg = sum_(ppc_500k["FG"]["power"])
    power_500k_chopin = sum_(ppc_500k["C"]["power"])

    lat_500k_chipletizer = sum_(ppc_500k["CP"]["perf"])
    lat_500k_rf = sum_(ppc_500k["RF"]["perf"])
    lat_500k_fg = sum_(ppc_500k["FG"]["perf"])
    lat_500k_chopin = sum_(ppc_500k["C"]["perf"])

    power_10M_chipletizer = sum_(ppc_10M["CP"]["power"])
    power_10M_rf = sum_(ppc_10M["RF"]["power"])
    power_10M_fg = sum_(ppc_10M["FG"]["power"])
    power_10M_chopin = sum_(ppc_10M["C"]["power"])

    lat_10M_chipletizer = sum_(ppc_10M["CP"]["perf"])
    lat_10M_rf = sum_(ppc_10M["RF"]["perf"])
    lat_10M_fg = sum_(ppc_10M["FG"]["perf"])
    lat_10M_chopin = sum_(ppc_10M["C"]["perf"])

    print("rf:%.2f" % ((power_500k_rf / power_500k_chipletizer + power_10M_rf / power_10M_chipletizer) / 2))
    print("rf:%.2f" % ((lat_500k_rf / lat_500k_chipletizer + lat_10M_rf / lat_10M_chipletizer) / 2))

    print("fg:%.2f" % ((power_500k_fg / power_500k_chipletizer + power_10M_fg / power_10M_chipletizer) / 2))
    print("fg:%.2f" % ((lat_500k_fg / lat_500k_chipletizer + lat_10M_fg / lat_10M_chipletizer) / 2))

    print("chopin:%.2f" % ((power_500k_chopin / power_500k_chipletizer + power_10M_chopin / power_10M_chipletizer) / 2))
    print("chopin:%.2f" % ((lat_500k_chopin / lat_500k_chipletizer + lat_10M_chopin / lat_10M_chipletizer) / 2))

    print("chipletizer:%.2f" % ((1 - power_10M_chipletizer / power_500k_chipletizer) * 100))
    print("chipletizer:%.2f" % ((1 - lat_10M_chipletizer / lat_500k_chipletizer) * 100))

    print(ppc_500k["CP"]["power"])


if __name__ == "__main__":
    power_comparison()