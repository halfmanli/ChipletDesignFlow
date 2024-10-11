# import cplex
import networkx as nx
from collections import defaultdict
import tempfile
import os
import shutil


class CType:
    """
        Chiplet type
    """
    def __init__(self, name, w, h, power, cost, pins, bw_pin, rsc_core, rsc_mem):
        """
            w/h: unit is mm
            bw_pin: input or output bandwdith of each pin
            rsc_core: core number
            rsc_mem: memory size, unit is GB
        """
        self.name = name
        self.w = w
        self.h = h
        self.area = w * h
        self.power = power
        self.cost = cost
        self.pins = pins
        self.bw_pin = bw_pin
        self.bw = len(pins) * bw_pin
        self.rsc_core = rsc_core
        self.rsc_mem = rsc_mem


class CInst:
    """
        Chiplet instance
    """
    def __init__(self, idx, ctype):
        self.idx = idx
        self.ctype = ctype


def select_chiplet(dir_cplex, acg, C, cfg, clean=True):
    """
        acg: application characterization graph
        C: candidate chiplet set
    """
    area = cfg["area"]
    k_area = cfg["k_area"]
    power = cfg["power"]
    k_power = cfg["k_power"]
    cost = cfg["cost"]
    k_cost = cfg["k_cost"]
    t_end = cfg["t_end"]
    k_t_end = cfg["k_t_end"]
    l_c2c = cfg["l_c2c"]

    obj = "Minimize {} power + {} area + {} cost + {} t_end\n".format(k_power, k_area, k_cost, k_t_end)
    st = "Subject To\n"
    bi = "Binary\n"

    # x_i_m
    bi += " ".join(["x_{}_{}".format(i, m) for i in acg for m in range(len(C))])
    for i in acg:
        x_i_m_avail = ["x_{}_{}".format(i, m) for m, ct in enumerate(C) if ct.ctype in acg.nodes[i]["C_avail"]]
        x_i_m_unavail = ["x_{}_{}".format(i, m) for m, ct in enumerate(C) if ct.ctype not in acg.nodes[i]["C_avail"]]
        assert x_i_m_avail  # available chiplet set is empty
        st += " + ".join(map(str, x_i_m_avail)) + " = 1\n"
        if x_i_m_unavail:
            st += " + ".join(map(str, x_i_m_unavail)) + " = 0\n"

    # y_i_j_m_n
    for i in acg:
        for j in acg:
            for m in range(len(C)):
                for n in range(len(C)):
                    st += "x_{0}_{2} + x_{1}_{3} - y_{0}_{1}_{2}_{3} <= 1\n".format(i, j, m, n)
                    st += "x_{0}_{2} + x_{1}_{3} - 2 y_{0}_{1}_{2}_{3} >= 0\n".format(i, j, m, n)
                    bi += " y_{}_{}_{}_{}".format(i, j, m, n)

    # task end time constraint
    # task j depends on task i
    for i, j in acg.edges():
        st += "T_{} + ".format(i)
        t_c2c = []  # time of chiplet to chiplet communication
        for m in range(len(C)):
            for n in range(len(C)):
                if m != n:  # communications between different chiplets have latency
                    t_c2c.append("{} y_{}_{}_{}_{}".format(l_c2c, i, j, m, n))
        st += " + ".join(t_c2c)
        t_exc = []
        for m in range(len(C)):
            t_exc.append("{} x_{}_{}".format(acg.nodes[j]["t_exc"][C[m].ctype], j, m))
        st += " + " + " + ".join(t_exc)
        st += "- T_{} <= 0\n".format(j)

    # task i is source
    for i in acg.nodes():
        if acg.in_degree(i) == 0:
            t_exc = []
            for m in range(len(C)):
                t_exc.append("{} x_{}_{}".format(acg.nodes[i]["t_exc"][C[m].ctype], i, m))
            st += " + ".join(t_exc)
            st += "- T_{} <= 0\n".format(i)

    # all t_end should earlier than deadline
    for i in acg.nodes():
        st += "T_{} <= {}\n".format(i, t_end)

    # core number constraint
    for m in range(len(C)):
        st += " + ".join(["{} x_{}_{}".format(acg.nodes[i]["rsc_core"][C[m].ctype], i, m) for i in acg])
        st += " <= {}\n".format(C[m].ctype.rsc_core)

    # memory constraint
    for m in range(len(C)):
        st += " + ".join(["{} x_{}_{}".format(acg.nodes[i]["rsc_mem"][C[m].ctype], i, m) for i in acg])
        st += " <= {}\n".format(C[m].ctype.rsc_mem)

    # communication bandwidth constraint
    # output
    for m in range(len(C)):
        st_bo = [
            "{} y_{}_{}_{}_{}".format(acg.edges[(i, j)]["comm"], i, j, m, n) for i, j in acg.edges() for n in range(len(C))
            if n != m
        ]
        st_bo = " + ".join(st_bo)
        st_bo += " <= {}\n".format(C[m].ctype.bw)
        st += st_bo
    # input
    for m in range(len(C)):
        st_bi = [
            "{} y_{}_{}_{}_{}".format(acg.edges[(i, j)]["comm"], i, j, n, m) for i, j in acg.edges() for n in range(len(C))
            if n != m
        ]
        st_bi = " + ".join(st_bi)
        st_bi += " <= {}\n".format(C[m].ctype.bw)
        st += st_bi

    # u_m
    for m in range(len(C)):
        bi += " u_{}".format(m)
        st_u_m = "".join(["u_{} - x_{}_{} >= 0\n".format(m, i, m) for i in acg])
        st_u_m += "u_{} - ".format(m) + " - ".join(["x_{}_{}".format(i, m) for i in acg]) + " <= 0\n"
        st += st_u_m

    # power
    st += "power - " + " - ".join(["{} u_{}".format(C[m].ctype.power, m) for m in range(len(C))]) + " = 0\n"
    st += "power <= {}\n".format(power)

    # area
    st += "area - " + " - ".join(["{} u_{}".format(C[m].ctype.area, m) for m in range(len(C))]) + " = 0\n"
    st += "area <= {}\n".format(area)

    # cost
    st += "cost - " + " - ".join(["{} u_{}".format(C[m].ctype.cost, m) for m in range(len(C))]) + " = 0\n"
    st += "cost <= {}\n".format(cost)

    lp = obj + st + bi + "\nEnd"
    dir_tmp = tempfile.mkdtemp(dir=os.path.join(dir_cplex, "build"))
    path_lp = os.path.join(dir_tmp, "solve.lp")
    with open(path_lp, "w") as f:
        f.write(lp)

    cpx = cplex.Cplex(path_lp)
    cpx.set_log_stream(None)
    cpx.set_error_stream(None)
    cpx.set_warning_stream(None)
    cpx.set_results_stream(None)
    cpx.solve()

    sol = {"assignment": {}}
    for i in acg:
        for m in range(len(C)):
            x_i_m = "x_{}_{}".format(i, m)
            sol["assignment"][x_i_m] = int(cpx.solution.get_values(x_i_m))
    sol["power"] = cpx.solution.get_values("power")
    sol["area"] = cpx.solution.get_values("area")
    sol["cost"] = cpx.solution.get_values("cost")
    sol["chiplets"] = []
    for m in range(len(C)):
        if int(cpx.solution.get_values("u_{}".format(m))):
            sol["chiplets"].append(m)

    if clean:
        shutil.rmtree(dir_tmp)


if __name__ == "__main__":
    cpu1 = CType(name="cpu1", w=12, h=12, power=10, cost=2, pins=[(0, 0), (2, 2)], bw_pin=16, rsc_core=15, rsc_mem=18)
    cpu2 = CType(name="cpu2", w=10, h=10, power=8, cost=1, pins=[(1, 1), (2, 2)], bw_pin=16, rsc_core=14, rsc_mem=19)

    C = [CInst(idx=0, ctype=cpu1), CInst(idx=2, ctype=cpu1), CInst(idx=3, ctype=cpu2)]

    acg = nx.DiGraph()  # Application Characterization Graph
    acg.add_node(0)
    acg.add_node(1)
    acg.add_node(2)
    acg.add_edge(0, 1, comm=16)
    acg.add_edge(0, 2, comm=15)
    acg.nodes[0]["C_avail"] = [cpu1]
    acg.nodes[1]["C_avail"] = [cpu1, cpu2]
    acg.nodes[2]["C_avail"] = [cpu1]

    acg.nodes[0]["t_exc"] = defaultdict(int, {cpu1: 1, cpu2: 2})
    acg.nodes[0]["rsc_core"] = defaultdict(int, {cpu1: 3, cpu2: 4})
    acg.nodes[0]["rsc_mem"] = defaultdict(int, {cpu1: 5, cpu2: 6})

    acg.nodes[1]["t_exc"] = defaultdict(int, {cpu1: 7, cpu2: 8})
    acg.nodes[1]["rsc_core"] = defaultdict(int, {cpu1: 9, cpu2: 10})
    acg.nodes[1]["rsc_mem"] = defaultdict(int, {cpu1: 11, cpu2: 12})

    acg.nodes[2]["t_exc"] = defaultdict(int, {cpu1: 13, cpu2: 14})
    acg.nodes[2]["rsc_core"] = defaultdict(int, {cpu1: 15, cpu2: 16})
    acg.nodes[2]["rsc_mem"] = defaultdict(int, {cpu1: 17, cpu2: 18})

    cfg = dict(area=2500, k_area=1, power=100, k_power=1, cost=10, k_cost=1, t_end=100, k_t_end=1, l_c2c=10)

    dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
    dir_cplex = os.path.join(dir_root, "tool/cplex")
    select_chiplet(dir_cplex=dir_cplex, acg=acg, C=C, cfg=cfg, clean=False)
