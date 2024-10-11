from enum import Enum
import networkx as nx
import itertools as itl
from bidict import bidict
from collections import defaultdict


class Direction(Enum):
    N = 0
    S = 1
    W = 2
    E = 3


def get_DIR(u, v):
    """
        return the direction pointing from u to v
    """
    x_u, y_u = u
    x_v, y_v = v
    assert abs(x_v - x_u) + abs(y_v - y_u) == 1
    if x_u == x_v:
        if y_u > y_v:
            return Direction.S
        else:
            return Direction.N
    else:
        assert y_u == y_v
        if x_u > x_v:
            return Direction.W
        else:
            return Direction.E


def get_mid(values):
    vals = sorted(values)
    idx_max = len(vals) - 1
    if len(vals) % 2:  # odd
        return vals[idx_max // 2]
    else:  # even
        return (vals[idx_max // 2] + vals[idx_max // 2 + 1]) // 2


def get_ppr(x_c, y_c, r):
    """
        Get list of (x, y) of peripheral of square whose center is (x_c, y_c) 

        r: radius = 1:
            x x x
            x o x
            x x x
    """
    sz = 2 * r + 1  # size of square
    ppr = list(
        itl.chain(itl.product([0], range(sz - 1)), itl.product(range(sz - 1), [sz - 1]), itl.product([sz - 1], range(1, sz)),
                  itl.product(range(1, sz), [0])))
    if r == 0:
        ppr = [(x_c, y_c)]
    else:
        ppr = [(x - r + x_c, y - r + y_c) for x, y in ppr]
    return ppr


def check_tile(csys, intp_graph, topo_graph, xy_rtr, xy_pin, x_t, y_t, rtr, cr=2):
    """
        Check the tile router, return True if can be mapped as router rtr in topology graph
        x_t, y_t: xy coordinates of considered tile
        xy_rtr: xy of routers have been mapped
        xy_pin: xy of pins/NIs
        x_t, y_t: the tile to check
        rtr: the router to be mapped
        r: search radius
    """
    for x_rel in range(-cr, cr + 1):
        for y_rel in range(-cr, cr + 1):
            x = x_t + x_rel
            y = y_t + y_rel
            if x < 0 or x >= csys.W or y < 0 or y >= csys.H:  # out of interposer
                continue
            if (x, y) in xy_rtr.inverse:  # has been set as rtr
                return False

    # check available channels
    # if the tile is connected to one NI/pin, then it can NOT be mapped as router of the NI/pin connected in topology graph
    #   because we set the core input channel as bypass channel
    cnted_pin = xy_pin.inverse[(x_t, y_t)] if (x_t, y_t) in xy_pin.inverse else None
    neigh_pins = [p for p in set(nx.all_neighbors(topo_graph, rtr)) if p < len(csys.task_graph)]
    if cnted_pin is None or cnted_pin not in neigh_pins:
        ic = sum([e_attr["nc"]
                  for (_, __, e_attr) in intp_graph.in_edges((x_t, y_t), data=True)])  # available input normal channel
        oc = sum([e_attr["nc"] for (_, __, e_attr) in intp_graph.out_edges((x_t, y_t), data=True)])
        return (topo_graph.in_degree(rtr) <= ic) and (topo_graph.out_degree(rtr) <= oc)
    else:
        return False


def map_router(csys, topo_graph, intp_graph, xy_rtr, xy_pin, rtr):
    """
        Map the router to logical tile
    """
    pnum = len(csys.task_graph)
    neigh_pins = [n for n in set(nx.all_neighbors(topo_graph, rtr)) if n < pnum]
    assert len(neigh_pins) > 0  # can not handle router only connected to routers
    xy_neighp = [xy_pin[p] for p in neigh_pins]
    x_neighp, y_neighp = list(zip(*xy_neighp))
    x_mid = get_mid(x_neighp)
    y_mid = get_mid(y_neighp)

    sr = 0  # search radius
    x_rtr, y_rtr = x_mid, y_mid
    cr = 3  # check radius
    while True:
        ppr = get_ppr(x_c=x_rtr, y_c=y_rtr, r=sr)
        for x, y in ppr:
            if (x < 1) or (x > csys.W - 2) or (y < 1) or (y > csys.H - 2):  # avoid boundary
                continue
            if check_tile(csys=csys,
                          intp_graph=intp_graph,
                          topo_graph=topo_graph,
                          xy_rtr=xy_rtr,
                          xy_pin=xy_pin,
                          x_t=x,
                          y_t=y,
                          rtr=rtr,
                          cr=cr):
                return (x, y)
        sr += 1
        if sr >= csys.W and sr >= csys.H:
            cr -= 1
            sr = 0
        assert cr >= 0


def heur(uv, pq):
    """
        Function to evaluate the estimate of the distance from the a node to the target.
        The function takes two nodes arguments and must return a number.

        uv, vw: edge node
    """
    u, _, _ = uv
    p, _, _ = pq
    if u[0] == "vs":  # for virtual start, u is ("vs", (x_src, y_src)); for others, uv is (x_u, y_u)
        x_s, y_s = u[2]
    else:
        x_s, y_s = u
    x_d, y_d = p

    return abs(x_d - x_s) + abs(y_d - y_s)  # Manhattan distance


def weight_func(u, v, e_attr):
    """
        The function must accept exactly three positional arguments:
            the two endpoints of an edge and the dictionary of edge attributes for that edge.
        The function must return a number.
    """
    return e_attr["b"] * e_attr["h"] * e_attr["p"] + e_attr["bend"]  # cost function


def find_path(intp_graph, sd, xy_rtr, base_cost, bend_cost, max_retry):
    """
        Find path from sources to destinations using A* searching algorithm.
        Note this algorithm will ignore the links used by previous src/dst
        
        intp_graph: grid-like graph map, nx directed graph
        sd: ordered list of ((x_src, y_src), (x_dst, y_dst))
        base_cost: base cost of using a link
        bend_cost: cost of bend/turn
    """
    if not intp_graph.is_directed():
        raise TypeError("ERROR: interposer graph should be directed")
    ig = intp_graph.copy()
    vs_edges = []  # virtual source nodes
    vd_edges = []  # virtual target nodes
    for src, dst in sd:  # src: (x, y, "c" or "r")
        x_src, y_src, t_src = src
        x_dst, y_dst, t_dst = dst
        vs_edges.append((("vs", t_src, (x_src, y_src)), (x_src, y_src)))  # add virtual source node ("vs", src)
        vd_edges.append(((x_dst, y_dst), ("vd", t_dst, (x_dst, y_dst))))  # add virtual target node ("vd", dst)
    ig.add_edges_from(vs_edges)
    ig.add_edges_from(vd_edges)
    assert len(vs_edges) == len(vd_edges)

    lg = nx.DiGraph()
    for f_e in ig.edges():  # from_edge node
        for t_e in ig.edges(f_e[1]):  # to_edge node
            if f_e[0][0] == "vs" and t_e[1][0] == "vd":
                continue
            # f_e: (u, v); t_e: (v, w)
            x_v, y_v = f_e[1]
            attr_f_e = ig.edges[f_e]
            attr_t_e = ig.edges[t_e]
            if (x_v, y_v) not in xy_rtr.inverse:  # v is not a router
                if f_e in vs_edges:
                    assert f_e[0][1] == "c"
                    if attr_t_e["nc"]:
                        lg.add_edge((*f_e, 1), (*t_e, 0))
                    if attr_t_e["bc"]:
                        lg.add_edge((*f_e, 1), (*t_e, 1))
                elif t_e in vd_edges:
                    assert t_e[1][1] == "c"
                    if attr_f_e["nc"]:
                        lg.add_edge((*f_e, 0), (*t_e, 1))
                    if attr_f_e["bc"]:
                        lg.add_edge((*f_e, 1), (*t_e, 1))
                else:
                    if attr_f_e["nc"] and attr_t_e["nc"]:
                        lg.add_edge((*f_e, 0), (*t_e, 0))
                    if attr_f_e["nc"] and attr_t_e["bc"]:
                        lg.add_edge((*f_e, 0), (*t_e, 1))
                    if attr_f_e["bc"] and attr_t_e["nc"]:
                        lg.add_edge((*f_e, 1), (*t_e, 0))
                    if attr_f_e["bc"] and attr_t_e["bc"]:
                        lg.add_edge((*f_e, 1), (*t_e, 1))

            else:  # v is a router  TODO: this is wrong, no need to ban turn from bc/nc to nc/bc with if attr_f_e["nc"] and attr_t_e["nc"]:
                if f_e in vs_edges:
                    if f_e[0][1] == "c":
                        if attr_t_e["bc"]:
                            lg.add_edge((*f_e, 1), (*t_e, 1))
                    else:
                        assert f_e[0][1] == "r"
                        if attr_t_e["nc"]:
                            lg.add_edge((*f_e, 1), (*t_e, 0))  # we assum virtual destination router using channel "1"
                elif t_e in vd_edges:
                    if t_e[1][1] == "c":
                        if attr_f_e["bc"]:
                            lg.add_edge((*f_e, 1), (*t_e, 1))
                    else:
                        assert t_e[1][1] == "r"
                        if attr_f_e["nc"]:
                            lg.add_edge((*f_e, 0), (*t_e, 1))
                else:
                    if attr_f_e["nc"] and attr_t_e["nc"]:
                        lg.add_edge((*f_e, 0), (*t_e, 0))
                    if attr_f_e["bc"] and attr_t_e["bc"]:
                        lg.add_edge((*f_e, 1), (*t_e, 1))

    nx.set_edge_attributes(lg, base_cost, name="b")  # base cost
    nx.set_edge_attributes(lg, 1, name="h")  # historical congestion
    nx.set_edge_attributes(lg, 1, name="capacity")  # link capacity

    # special treatment to source edges and bends(turns)
    for (u, v, cha_uv), (_, w, cha_vw), e_attr in lg.edges(data=True):
        if u[0] == "vs":  # erase base cost for virtual source edges
            e_attr["b"] = 0

        if u[0] != "vs" and w[0] != "vd" and (get_DIR(u, v) != get_DIR(v, w) or cha_uv != cha_vw):  # is a turn
            e_attr["bend"] = bend_cost
        else:
            e_attr["bend"] = 0

    h_fac = 0.5  # constant for all iterations; between 0.2 and 1 works well
    p_fac = 0.5  # increase by 1.5 to 2 times each iteration
    for _ in range(max_retry):
        # generate line graph
        nx.set_edge_attributes(lg, 1, name="p")  # present congestion
        nx.set_edge_attributes(lg, 0, name="occupancy")  # all nets are unused
        paths = []
        used_links = []
        for se, de in zip(vs_edges, vd_edges):  # for every net
            # for astar, we dynamically calculate the weight
            path = nx.astar_path(G=lg, source=(*se, 1), target=(*de, 1), heuristic=heur,
                                 weight=weight_func)  # [vs_edge, edge 0, ..., dst_edge]
            paths.append(path[1:-1])
            # update present congestion
            for e_m in path[1:-1]:  # no need to consider virtual source/destination
                used_links.append(e_m)
                for e_mn in lg.edges(e_m):
                    e_attr = lg.edges[e_mn]
                    occ = e_attr["occupancy"] + 1
                    e_attr["occupancy"] = occ  # update usage of link
                    # update p_n: p_n = 1 + max(0, [occupancy(n) + 1 - capacity(n)] * p_fac)
                    e_attr["p"] = 1 + max(0, (occ + 1 - e_attr["capacity"]) * p_fac)
        p_fac *= 1.5

        # historical congestion penalty is updated only after an entire routing iteration
        for e_m in set(used_links):
            for e_mn in lg.edges(e_m):
                # h(n)^i = h(n)^(i-1) + max(0, [occupancy(n) - capacity(n)] * h_fac)
                e_attr = lg.edges[e_mn]
                e_attr["h"] += max(0, (e_attr["occupancy"] - e_attr["capacity"]) * h_fac)

        flg_suc = True
        # print(len(used_links) - len((set(used_links))), max([(l, used_links.count(l)) for l in used_links], key=lambda v: v[1]))
        for _, _, e_attr in lg.edges(data=True):
            if e_attr["occupancy"] > e_attr["capacity"]:  # illegal routing
                flg_suc = False
                break
        if flg_suc:
            return True, paths
    return False, paths


def gen_layout_active(csys, topo_graph, rtp, placement, faulted_links=None, base_cost=1, bend_cost=0, max_retry=90):
    """
        Manhattan Layout Generation

        Return: xy_rtr, paths

        topo_graph: nx directed graph, node names should be 0, 1, 2, ...
        rtp: routing path list dict: rtp[(src, dst)] = [rtr_0, rtr_1, ...]
        placement: list of triad: (cx, cy, angle), same order as chiplets; cx, cy: bottom left corner of chiplet
        faulted_links: list of links to be removed,foramt is [((x_0, y_0), (x_1, y_1)),((x_2, y_2), (x_3, y_3)), ...]
    """
    if not topo_graph.is_directed():
        raise TypeError("ERROR: task graph and topology graph should be directed")

    # build the grpah of interposer, grid/mesh
    intp_graph = nx.DiGraph()
    for (x_u, x_v) in zip(range(0, csys.W), range(1, csys.W)):
        for y in range(csys.H):
            intp_graph.add_edge((x_u, y), (x_v, y), nc=1, bc=1)  # nc, bc: normal/bypass channel
            intp_graph.add_edge((x_v, y), (x_u, y), nc=1, bc=1)

    for (y_u, y_v) in zip(range(0, csys.H), range(1, csys.H)):
        for x in range(csys.W):
            intp_graph.add_edge((x, y_u), (x, y_v), nc=1, bc=1)
            intp_graph.add_edge((x, y_v), (x, y_u), nc=1, bc=1)

    xy_pin = bidict()  # get the position of pin/NI
    for p in csys.task_graph:
        x_p, y_p = csys.get_xy_pin(p, placement)
        xy_pin[p] = (x_p, y_p)
        e_Si = ((x_p, y_p - 1), (x_p, y_p))
        e_So = ((x_p, y_p), (x_p, y_p - 1))
        if csys.task_graph.in_degree(p) and intp_graph.has_edge(
                *e_So):  # The South input and output port is multiplexed with Loacal port
            assert intp_graph.edges[e_So]["bc"] == 1
            intp_graph.edges[e_So]["bc"] -= 1
        if csys.task_graph.out_degree(p) and intp_graph.has_edge(*e_Si):
            assert intp_graph.edges[e_Si]["bc"] == 1
            intp_graph.edges[e_Si]["bc"] -= 1

    if faulted_links is not None:
        for (u, v, t) in faulted_links:  # t: type of channel
            assert t == "nc" or t == "bc"
            intp_graph[u][v][t] = 0

    pnum = len(csys.task_graph)
    comm_r = defaultdict(int)  # sum of input=output communication traffic, key is router
    comm_l = defaultdict(int)  # accumulated communication traffic, key is topological link
    for s, d, e_attr in csys.task_graph.edges(data=True):  # for every flow from pin s to d
        p = [s] + rtp[(s, d)] + [d]
        for u, v in zip(p, p[1:]):
            comm_l[(u, v)] += e_attr["comm"]
            if v >= pnum:  # add input traffic to routers
                comm_r[v] += e_attr["comm"]
    assert set(topo_graph.edges()) == set(comm_l.keys())  # Note all the links in topo_graph should be used in rtp

    xy_rtr = bidict()  # key is router id
    for rtr, _ in sorted(comm_r.items(), key=lambda kv: kv[1], reverse=True):  # begin from the busiest router
        pos = map_router(csys=csys, topo_graph=topo_graph, intp_graph=intp_graph, xy_rtr=xy_rtr, xy_pin=xy_pin, rtr=rtr)
        xy_rtr[rtr] = pos

    sd_pr = [kv[0] for kv in sorted(comm_l.items(), key=lambda kv: kv[1], reverse=True)]  # src/dst of pin&router
    xy_pr = dict(list(xy_pin.items()) + list(xy_rtr.items()))
    sd = []
    for s, d in sd_pr:
        if s < pnum:
            s_xy = xy_pr[s] + ("c", )  # use "c" and "r" to distinguish core/NI/pin and router
        else:
            s_xy = xy_pr[s] + ("r", )

        if d < pnum:
            d_xy = xy_pr[d] + ("c", )
        else:
            d_xy = xy_pr[d] + ("r", )

        sd.append((s_xy, d_xy))  # map pins and routers to tile

    flg_suc, paths_ = find_path(intp_graph=intp_graph,
                                sd=sd,
                                xy_rtr=xy_rtr,
                                base_cost=base_cost,
                                bend_cost=bend_cost,
                                max_retry=max_retry)
    paths = {}
    for (s, d), path in zip(sd_pr, paths_):
        paths[(s, d)] = path

    return flg_suc, xy_pr, paths