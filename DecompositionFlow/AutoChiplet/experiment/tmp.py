import pickle

for q in [500000, 10000000]:
    with open("/research/Chipletization/log/exp_1/{}/res_raw.pickle".format(q), "rb") as f:
        ppc, cd = pickle.load(f)

    ppc_new = {}
    cd_new = {}

    for strategy in ["CP", "M", "RF", "BP", "FG", "C"]:
        ppc_new[strategy] = {}
        cd_new[strategy] = []
        for ind in ["power", "perf", "cost"]:
            ppc_new[strategy][ind] = []
            for i in [1, 2, 3, 0, 4]:
                ppc_new[strategy][ind].append(ppc[strategy][ind][i])
                cd_new[strategy].append(cd[strategy][i])

    res = [ppc_new, cd_new]
    with open("/research/Chipletization/log/exp_1/{}/res.pickle".format(q), "wb") as f:
        pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)