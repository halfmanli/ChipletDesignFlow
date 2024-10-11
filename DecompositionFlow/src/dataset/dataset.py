from typing import List

import networkx as nx

from ..model import Block, spec


def check_blocks(blks):
    """
        Check duplicated blocks
    """
    return len(blks) == len(set(blks))


def check_bdgs(bdgs):
    """
        Check bdg list
    """
    for idx_bdg, bdg in enumerate(bdgs):
        nodes = list(bdg.nodes())
        if set(nodes) != set(range(len(nodes))):
            print("ERROR: {} owns incorrect nodes.".format(idx_bdg))
            return False
    return True


def add_edge_m2m(bdg: nx.DiGraph, src_nodes: List[int], dst_blocks: List[Block], vol_sum: float):
    """
        add edge: many to many
        vol_sum: input + output
    """
    dst_nodes = []
    for n, attr_n in bdg.nodes(data=True):
        if attr_n["block"] in dst_blocks:
            dst_nodes.append(n)

    for i in src_nodes:
        comm = vol_sum / len(dst_nodes) / 2
        for j in dst_nodes:
            bdg.add_edge(i, j, comm=comm, ener_eff=spec.D2D_energy_efficiency, perf_penal=1)
            bdg.add_edge(j, i, comm=comm, ener_eff=spec.D2D_energy_efficiency, perf_penal=1)


def add_edge_b2b(bdg: nx.DiGraph, block: Block, vol_sum: float):
    nodes = []
    for n, attr_n in bdg.nodes(data=True):
        if attr_n["block"] == block:
            nodes.append(n)

    for i in range(len(nodes)):
        comm = vol_sum / (len(nodes) - 1) / 2
        for j in range(i + 1, len(nodes)):
            bdg.add_edge(nodes[i], nodes[j], comm=comm, ener_eff=spec.D2D_energy_efficiency, perf_penal=1)
            bdg.add_edge(nodes[j], nodes[i], comm=comm, ener_eff=spec.D2D_energy_efficiency, perf_penal=1)


def add_edge_c2c(bdg: nx.DiGraph, cores: List[int], caches: List[int], comm: float):
    assert len(cores) % len(caches) == 0
    step = len(cores) // len(caches)
    for i in range(len(caches)):
        for j in range(step):
            bdg.add_edge(caches[i], cores[i * step + j], comm=comm / 2, ener_eff=spec.D2D_energy_efficiency, perf_penal=1)
            bdg.add_edge(cores[i * step + j], caches[i], comm=comm / 2, ener_eff=spec.D2D_energy_efficiency, perf_penal=1)


def AMD():
    """
        Epyc Rome
        CCX = 1 L3 + 4 core
    """
    core = Block(name="core", area=5.05, node=7)
    L3 = Block(name="L3", area=16.8, node=7)  # 16 MB L3
    ddr = Block(name="ddr", area=33.74, node=14)  # dual channel
    pcie = Block(name="pcie", area=35.26, node=14)  # x16 64 GB/s

    assert check_blocks([core, L3, ddr, pcie])

    epyc_7282 = nx.DiGraph(name="epyc_7282")
    epyc_7282.add_nodes_from(range(0, 16), block=core)
    epyc_7282.add_nodes_from(range(16, 20), block=L3)
    epyc_7282.add_nodes_from(range(20, 24), block=ddr)
    epyc_7282.add_nodes_from(range(24, 32), block=pcie)
    add_edge_c2c(bdg=epyc_7282, cores=list(range(0, 16)), caches=list(range(16, 20)), comm=64)
    add_edge_m2m(bdg=epyc_7282, src_nodes=range(20, 24), dst_blocks=[L3], vol_sum=51.2)
    add_edge_m2m(bdg=epyc_7282, src_nodes=range(24, 32), dst_blocks=[L3], vol_sum=64)
    add_edge_b2b(bdg=epyc_7282, block=L3, vol_sum=64)

    epyc_7502p = nx.DiGraph(name="epyc_7502p")
    epyc_7502p.add_nodes_from(range(0, 32), block=core)
    epyc_7502p.add_nodes_from(range(32, 40), block=L3)
    epyc_7502p.add_nodes_from(range(40, 44), block=ddr)
    epyc_7502p.add_nodes_from(range(44, 52), block=pcie)
    add_edge_c2c(bdg=epyc_7502p, cores=list(range(0, 32)), caches=list(range(32, 40)), comm=64)
    add_edge_m2m(bdg=epyc_7502p, src_nodes=range(40, 44), dst_blocks=[L3], vol_sum=51.2)
    add_edge_m2m(bdg=epyc_7502p, src_nodes=range(44, 52), dst_blocks=[L3], vol_sum=64)
    add_edge_b2b(bdg=epyc_7502p, block=L3, vol_sum=64)

    epyc_7552 = nx.DiGraph(name="epyc_7552")
    epyc_7552.add_nodes_from(range(0, 48), block=core)
    epyc_7552.add_nodes_from(range(48, 60), block=L3)
    epyc_7552.add_nodes_from(range(60, 64), block=ddr)
    epyc_7552.add_nodes_from(range(64, 72), block=pcie)
    add_edge_c2c(bdg=epyc_7552, cores=list(range(0, 48)), caches=list(range(48, 60)), comm=64)
    add_edge_m2m(bdg=epyc_7552, src_nodes=range(60, 64), dst_blocks=[L3], vol_sum=51.2)
    add_edge_m2m(bdg=epyc_7552, src_nodes=range(64, 72), dst_blocks=[L3], vol_sum=64)
    add_edge_b2b(bdg=epyc_7552, block=L3, vol_sum=64)

    epyc_7702p = nx.DiGraph(name="epyc_7702p")
    epyc_7702p.add_nodes_from(range(0, 64), block=core)
    epyc_7702p.add_nodes_from(range(64, 80), block=L3)
    epyc_7702p.add_nodes_from(range(80, 84), block=ddr)
    epyc_7702p.add_nodes_from(range(84, 92), block=pcie)
    add_edge_c2c(bdg=epyc_7702p, cores=list(range(0, 64)), caches=list(range(64, 80)), comm=64)
    add_edge_m2m(bdg=epyc_7702p, src_nodes=range(80, 84), dst_blocks=[L3], vol_sum=51.2)
    add_edge_m2m(bdg=epyc_7702p, src_nodes=range(84, 92), dst_blocks=[L3], vol_sum=64)
    add_edge_b2b(bdg=epyc_7702p, block=L3, vol_sum=64)

    ryzen_7_3700x = nx.DiGraph(name="ryzen_7_3700x")
    ryzen_7_3700x.add_nodes_from(range(0, 8), block=core)
    ryzen_7_3700x.add_nodes_from(range(8, 10), block=L3)
    ryzen_7_3700x.add_nodes_from(range(10, 11), block=ddr)
    ryzen_7_3700x.add_nodes_from(range(11, 13), block=pcie)
    add_edge_c2c(bdg=ryzen_7_3700x, cores=list(range(0, 8)), caches=list(range(8, 10)), comm=64)
    add_edge_m2m(bdg=ryzen_7_3700x, src_nodes=range(10, 11), dst_blocks=[L3], vol_sum=51.2)
    add_edge_m2m(bdg=ryzen_7_3700x, src_nodes=range(11, 13), dst_blocks=[L3], vol_sum=64)
    add_edge_b2b(bdg=ryzen_7_3700x, block=L3, vol_sum=64)

    ryzen_9_3900 = nx.DiGraph(name="ryzen_9_3900")
    ryzen_9_3900.add_nodes_from(range(0, 16), block=core)
    ryzen_9_3900.add_nodes_from(range(16, 20), block=L3)
    ryzen_9_3900.add_nodes_from(range(20, 21), block=ddr)
    ryzen_9_3900.add_nodes_from(range(21, 23), block=pcie)
    add_edge_c2c(bdg=ryzen_9_3900, cores=list(range(0, 16)), caches=list(range(16, 20)), comm=64)
    add_edge_m2m(bdg=ryzen_9_3900, src_nodes=range(10, 11), dst_blocks=[L3], vol_sum=51.2)
    add_edge_m2m(bdg=ryzen_9_3900, src_nodes=range(11, 13), dst_blocks=[L3], vol_sum=64)
    add_edge_b2b(bdg=ryzen_9_3900, block=L3, vol_sum=64)

    bdg_all = [epyc_7282, epyc_7502p, epyc_7552, epyc_7702p, ryzen_7_3700x, ryzen_9_3900]
    assert check_bdgs(bdgs=bdg_all)

    return bdg_all


def Intel():
    sunny_cove = Block(name="sunny_cove", area=9.04, node=10)
    pcie = Block(name="pcie", area=7.82, node=10)  # 16 lanes
    upi = Block(name="upi", area=18.75, node=10)
    ddr = Block(name="ddr", area=27.88, node=10)  # 2 channels

    check_blocks(blks=[sunny_cove, pcie, upi, ddr])

    xeon_platinum_8380 = nx.DiGraph(name="xeon_platinum_8380")
    xeon_platinum_8380.add_nodes_from(range(0, 40), block=sunny_cove)
    xeon_platinum_8380.add_nodes_from(range(40, 44), block=pcie)
    xeon_platinum_8380.add_nodes_from(range(44, 47), block=upi)
    xeon_platinum_8380.add_nodes_from(range(47, 51), block=ddr)
    add_edge_m2m(bdg=xeon_platinum_8380, src_nodes=range(40, 44), dst_blocks=[sunny_cove], vol_sum=64)
    add_edge_m2m(bdg=xeon_platinum_8380, src_nodes=range(44, 47), dst_blocks=[sunny_cove], vol_sum=41.6)
    add_edge_m2m(bdg=xeon_platinum_8380, src_nodes=range(47, 51), dst_blocks=[sunny_cove], vol_sum=51.2)
    add_edge_b2b(bdg=xeon_platinum_8380, block=sunny_cove, vol_sum=64)

    xeon_platinum_8368 = nx.DiGraph(name="xeon_platinum_8368")
    xeon_platinum_8368.add_nodes_from(range(0, 38), block=sunny_cove)
    xeon_platinum_8368.add_nodes_from(range(38, 42), block=pcie)
    xeon_platinum_8368.add_nodes_from(range(42, 45), block=upi)
    xeon_platinum_8368.add_nodes_from(range(45, 49), block=ddr)
    add_edge_m2m(bdg=xeon_platinum_8368, src_nodes=range(38, 42), dst_blocks=[sunny_cove], vol_sum=64)
    add_edge_m2m(bdg=xeon_platinum_8368, src_nodes=range(42, 45), dst_blocks=[sunny_cove], vol_sum=41.6)
    add_edge_m2m(bdg=xeon_platinum_8368, src_nodes=range(45, 49), dst_blocks=[sunny_cove], vol_sum=51.2)
    add_edge_b2b(bdg=xeon_platinum_8368, block=sunny_cove, vol_sum=64)

    xeon_platinum_8362 = nx.DiGraph(name="xeon_platinum_8362")
    xeon_platinum_8362.add_nodes_from(range(0, 32), block=sunny_cove)
    xeon_platinum_8362.add_nodes_from(range(32, 36), block=pcie)
    xeon_platinum_8362.add_nodes_from(range(36, 39), block=upi)
    xeon_platinum_8362.add_nodes_from(range(39, 43), block=ddr)
    add_edge_m2m(bdg=xeon_platinum_8362, src_nodes=range(32, 36), dst_blocks=[sunny_cove], vol_sum=64)
    add_edge_m2m(bdg=xeon_platinum_8362, src_nodes=range(36, 39), dst_blocks=[sunny_cove], vol_sum=41.6)
    add_edge_m2m(bdg=xeon_platinum_8362, src_nodes=range(39, 43), dst_blocks=[sunny_cove], vol_sum=51.2)
    add_edge_b2b(bdg=xeon_platinum_8362, block=sunny_cove, vol_sum=64)

    xeon_platinum_8360y = nx.DiGraph(name="xeon_platinum_8360y")
    xeon_platinum_8360y.add_nodes_from(range(0, 36), block=sunny_cove)
    xeon_platinum_8360y.add_nodes_from(range(36, 40), block=pcie)
    xeon_platinum_8360y.add_nodes_from(range(40, 43), block=upi)
    xeon_platinum_8360y.add_nodes_from(range(43, 47), block=ddr)
    add_edge_m2m(bdg=xeon_platinum_8360y, src_nodes=range(36, 40), dst_blocks=[sunny_cove], vol_sum=64)
    add_edge_m2m(bdg=xeon_platinum_8360y, src_nodes=range(40, 43), dst_blocks=[sunny_cove], vol_sum=41.6)
    add_edge_m2m(bdg=xeon_platinum_8360y, src_nodes=range(43, 47), dst_blocks=[sunny_cove], vol_sum=51.2)
    add_edge_b2b(bdg=xeon_platinum_8360y, block=sunny_cove, vol_sum=64)

    xeon_gold_6348 = nx.DiGraph(name="xeon_gold_6348")
    xeon_gold_6348.add_nodes_from(range(0, 28), block=sunny_cove)
    xeon_gold_6348.add_nodes_from(range(28, 32), block=pcie)
    xeon_gold_6348.add_nodes_from(range(32, 35), block=upi)
    xeon_gold_6348.add_nodes_from(range(35, 39), block=ddr)
    add_edge_m2m(bdg=xeon_gold_6348, src_nodes=range(28, 32), dst_blocks=[sunny_cove], vol_sum=64)
    add_edge_m2m(bdg=xeon_gold_6348, src_nodes=range(32, 35), dst_blocks=[sunny_cove], vol_sum=41.6)
    add_edge_m2m(bdg=xeon_gold_6348, src_nodes=range(35, 39), dst_blocks=[sunny_cove], vol_sum=51.2)
    add_edge_b2b(bdg=xeon_gold_6348, block=sunny_cove, vol_sum=64)

    xeon_silver_4316 = nx.DiGraph(name="xeon_silver_4316")
    xeon_silver_4316.add_nodes_from(range(0, 20), block=sunny_cove)
    xeon_silver_4316.add_nodes_from(range(20, 24), block=pcie)
    xeon_silver_4316.add_nodes_from(range(24, 26), block=upi)
    xeon_silver_4316.add_nodes_from(range(26, 30), block=ddr)
    add_edge_m2m(bdg=xeon_silver_4316, src_nodes=range(20, 24), dst_blocks=[sunny_cove], vol_sum=64)
    add_edge_m2m(bdg=xeon_silver_4316, src_nodes=range(24, 26), dst_blocks=[sunny_cove], vol_sum=41.6)
    add_edge_m2m(bdg=xeon_silver_4316, src_nodes=range(26, 30), dst_blocks=[sunny_cove], vol_sum=51.2)
    add_edge_b2b(bdg=xeon_silver_4316, block=sunny_cove, vol_sum=64)

    xeon_silver_4314 = nx.DiGraph(name="xeon_silver_4314")
    xeon_silver_4314.add_nodes_from(range(0, 16), block=sunny_cove)
    xeon_silver_4314.add_nodes_from(range(16, 20), block=pcie)
    xeon_silver_4314.add_nodes_from(range(20, 22), block=upi)
    xeon_silver_4314.add_nodes_from(range(22, 26), block=ddr)
    add_edge_m2m(bdg=xeon_silver_4314, src_nodes=range(16, 20), dst_blocks=[sunny_cove], vol_sum=64)
    add_edge_m2m(bdg=xeon_silver_4314, src_nodes=range(20, 22), dst_blocks=[sunny_cove], vol_sum=41.6)
    add_edge_m2m(bdg=xeon_silver_4314, src_nodes=range(22, 26), dst_blocks=[sunny_cove], vol_sum=51.2)
    add_edge_b2b(bdg=xeon_silver_4314, block=sunny_cove, vol_sum=64)

    xeon_silver_4310t = nx.DiGraph(name="xeon_silver_4310t")
    xeon_silver_4310t.add_nodes_from(range(0, 10), block=sunny_cove)
    xeon_silver_4310t.add_nodes_from(range(10, 14), block=pcie)
    xeon_silver_4310t.add_nodes_from(range(14, 16), block=upi)
    xeon_silver_4310t.add_nodes_from(range(16, 20), block=ddr)
    add_edge_m2m(bdg=xeon_silver_4310t, src_nodes=range(10, 14), dst_blocks=[sunny_cove], vol_sum=64)
    add_edge_m2m(bdg=xeon_silver_4310t, src_nodes=range(14, 16), dst_blocks=[sunny_cove], vol_sum=41.6)
    add_edge_m2m(bdg=xeon_silver_4310t, src_nodes=range(16, 20), dst_blocks=[sunny_cove], vol_sum=51.2)
    add_edge_b2b(bdg=xeon_silver_4310t, block=sunny_cove, vol_sum=64)

    xeon_silver_4310 = nx.DiGraph(name="xeon_silver_4310")
    xeon_silver_4310.add_nodes_from(range(0, 12), block=sunny_cove)
    xeon_silver_4310.add_nodes_from(range(12, 16), block=pcie)
    xeon_silver_4310.add_nodes_from(range(16, 18), block=upi)
    xeon_silver_4310.add_nodes_from(range(18, 22), block=ddr)
    add_edge_m2m(bdg=xeon_silver_4310, src_nodes=range(12, 16), dst_blocks=[sunny_cove], vol_sum=64)
    add_edge_m2m(bdg=xeon_silver_4310, src_nodes=range(16, 18), dst_blocks=[sunny_cove], vol_sum=41.6)
    add_edge_m2m(bdg=xeon_silver_4310, src_nodes=range(18, 22), dst_blocks=[sunny_cove], vol_sum=51.2)
    add_edge_b2b(bdg=xeon_silver_4310, block=sunny_cove, vol_sum=64)

    xeon_silver_4309y = nx.DiGraph(name="xeon_silver_4309y")
    xeon_silver_4309y.add_nodes_from(range(0, 8), block=sunny_cove)
    xeon_silver_4309y.add_nodes_from(range(8, 12), block=pcie)
    xeon_silver_4309y.add_nodes_from(range(12, 14), block=upi)
    xeon_silver_4309y.add_nodes_from(range(14, 18), block=ddr)
    add_edge_m2m(bdg=xeon_silver_4309y, src_nodes=range(8, 12), dst_blocks=[sunny_cove], vol_sum=64)
    add_edge_m2m(bdg=xeon_silver_4309y, src_nodes=range(12, 14), dst_blocks=[sunny_cove], vol_sum=41.6)
    add_edge_m2m(bdg=xeon_silver_4309y, src_nodes=range(14, 18), dst_blocks=[sunny_cove], vol_sum=51.2)
    add_edge_b2b(bdg=xeon_silver_4309y, block=sunny_cove, vol_sum=64)

    bdg_all = [
        xeon_platinum_8380, xeon_platinum_8368, xeon_platinum_8362, xeon_platinum_8360y, xeon_gold_6348, xeon_silver_4316,
        xeon_silver_4314, xeon_silver_4310t, xeon_silver_4310, xeon_silver_4309y
    ]
    assert check_bdgs(bdgs=bdg_all)

    return bdg_all


def Rockchip():
    # cortex_a55 = Block(name="cortex_a55", area=5.41, node=22)
    cortex_a55 = Block(name="cortex_a55", area=8.61, node=28)
    cortex_a53 = Block(name="cortex_a53", area=8.61, node=28)
    cortex_a72 = Block(name="cortex_a72", area=15.27, node=28)
    cortex_a17 = Block(name="cortex_a17", area=1.93, node=28)
    cortex_a35 = Block(name="cortex_a35", area=4, node=28)

    # mali_g52 = Block(name="mali_g52", area=9.97, node=22)
    mali_g52 = Block(name="mali_g52", area=15.86, node=28)
    mali_t860 = Block(name="mali_t860", area=7.66, node=28)
    mali_t760 = Block(name="mali_t760", area=7.66, node=28)
    g6110 = Block(name="g6110", area=4.78, node=28)
    mali_g31 = Block(name="mali_g31", area=15.87, node=28)
    mali_450 = Block(name="mali_450", area=7.66, node=28)

    # ddr_ctrl = Block(name="ddr_ctrl", area=7.36, node=22)
    ddr_ctrl = Block(name="ddr_ctrl", area=11.7, node=28)

    # npu_0 = Block(name="npu_0", area=3.3, node=22)  # 0.8 TOPS
    # npu_1 = Block(name="npu_1", area=12.38, node=22)  # 3 TOPS
    npu_0 = Block(name="npu_0", area=5.25, node=28)  # 0.8 TOPS
    npu_1 = Block(name="npu_1", area=5.25, node=28)  # 512 MAC
    npu_2 = Block(name="npu_2", area=19.68, node=28)  # 3 TOPS

    vcodec = Block(name="vcodec", area=7.14, node=28)

    isp_0 = Block(name="isp_0", area=12.54, node=28)  # 8 M
    isp_1 = Block(name="isp_1", area=20.38, node=28)  # 13 M

    io_0 = Block(name="io_0", area=54.18, node=28)  # PCIE (2 lanes) + SATA (3 lanes) + Other
    io_1 = Block(name="io_1", area=40.94, node=28)  # PCIE + Other
    io_2 = Block(name="io_2", area=36.53, node=28)  # Other

    assert check_blocks(blks=[
        cortex_a55, cortex_a53, cortex_a72, cortex_a17, cortex_a35, mali_g52, mali_t860, mali_t760, g6110, mali_g31, mali_450,
        ddr_ctrl, npu_0, npu_1, npu_2, vcodec, isp_0, isp_1, io_0, io_1, io_2
    ])

    rk3568 = nx.DiGraph(name="rk3568")
    rk3568.add_nodes_from(range(0, 4), block=cortex_a55)
    rk3568.add_nodes_from(range(4, 6), block=mali_g52)
    rk3568.add_node(6, block=ddr_ctrl)
    rk3568.add_node(7, block=npu_0)
    rk3568.add_node(8, block=vcodec)
    rk3568.add_node(9, block=isp_0)
    rk3568.add_node(10, block=io_0)
    add_edge_m2m(bdg=rk3568, src_nodes=[6], dst_blocks=[cortex_a55, mali_g52, npu_0, vcodec, isp_0, io_0], vol_sum=25.6)

    rk3568j = nx.DiGraph(name="rk3568j")
    rk3568j.add_nodes_from(range(0, 4), block=cortex_a55)
    rk3568j.add_nodes_from(range(4, 6), block=mali_g52)
    rk3568j.add_node(6, block=ddr_ctrl)
    rk3568j.add_node(7, block=npu_1)
    rk3568j.add_node(8, block=vcodec)
    rk3568j.add_node(9, block=isp_0)
    rk3568j.add_node(10, block=io_0)
    add_edge_m2m(bdg=rk3568j, src_nodes=[6], dst_blocks=[cortex_a55, mali_g52, npu_1, vcodec, isp_0, io_0], vol_sum=25.6)

    rk3566 = nx.DiGraph(name="rk3566")
    rk3566.add_nodes_from(range(0, 4), block=cortex_a55)
    rk3566.add_nodes_from(range(4, 6), block=mali_g52)
    rk3566.add_node(6, block=ddr_ctrl)
    rk3566.add_node(7, block=npu_0)
    rk3566.add_node(8, block=vcodec)
    rk3566.add_node(9, block=isp_0)
    rk3566.add_node(10, block=io_1)
    add_edge_m2m(bdg=rk3566, src_nodes=[6], dst_blocks=[cortex_a55, mali_g52, npu_0, vcodec, isp_0, io_1], vol_sum=25.6)

    rk3399 = nx.DiGraph(name="rk3399")
    rk3399.add_nodes_from(range(0, 4), block=cortex_a53)
    rk3399.add_nodes_from(range(4, 6), block=cortex_a72)
    rk3399.add_nodes_from(range(6, 10), block=mali_t860)
    rk3399.add_nodes_from(range(10, 13), block=ddr_ctrl)
    rk3399.add_node(12, block=vcodec)
    rk3399.add_nodes_from(range(13, 15), block=isp_1)
    rk3399.add_node(15, block=io_1)
    add_edge_m2m(bdg=rk3399,
                 src_nodes=[10, 11],
                 dst_blocks=[cortex_a53, cortex_a72, mali_t860, vcodec, isp_1, io_1],
                 vol_sum=25.6)

    rk3399pro = nx.DiGraph(name="rk3399pro")
    rk3399pro.add_nodes_from(range(0, 4), block=cortex_a53)
    rk3399pro.add_nodes_from(range(4, 6), block=cortex_a72)
    rk3399pro.add_nodes_from(range(6, 10), block=mali_t860)
    rk3399pro.add_nodes_from(range(10, 13), block=ddr_ctrl)
    rk3399pro.add_node(13, block=npu_2)
    rk3399pro.add_node(14, block=vcodec)
    rk3399pro.add_nodes_from(range(15, 17), block=isp_1)
    rk3399pro.add_node(17, block=io_1)
    add_edge_m2m(bdg=rk3399pro,
                 src_nodes=[10, 12],
                 dst_blocks=[cortex_a53, cortex_a72, mali_t860, npu_2, vcodec, isp_1, io_1],
                 vol_sum=25.6)

    rk3288 = nx.DiGraph(name="rk3288")
    rk3288.add_nodes_from(range(0, 4), block=cortex_a17)
    rk3288.add_nodes_from(range(4, 8), block=mali_t760)
    rk3288.add_nodes_from(range(8, 10), block=ddr_ctrl)
    rk3288.add_node(10, block=vcodec)
    rk3288.add_node(11, block=isp_1)
    rk3288.add_node(12, block=io_2)
    add_edge_m2m(bdg=rk3288, src_nodes=[8, 9], dst_blocks=[cortex_a17, mali_t760, vcodec, isp_1, io_2], vol_sum=25.6)

    rk3368 = nx.DiGraph(name="rk3368")
    rk3368.add_nodes_from(range(0, 8), block=cortex_a53)
    rk3368.add_node(8, block=g6110)
    rk3368.add_node(9, block=ddr_ctrl)
    rk3368.add_node(10, block=vcodec)
    rk3368.add_node(11, block=isp_0)
    rk3368.add_node(12, block=io_2)
    add_edge_m2m(bdg=rk3368, src_nodes=[9], dst_blocks=[cortex_a53, g6110, vcodec, isp_0, io_2], vol_sum=25.6)

    rk3326 = nx.DiGraph(name="rk3326")
    rk3326.add_nodes_from(range(0, 4), block=cortex_a35)
    rk3326.add_nodes_from(range(4, 6), block=mali_g31)
    rk3326.add_node(6, block=ddr_ctrl)
    rk3326.add_node(7, block=vcodec)
    rk3326.add_node(8, block=isp_0)
    rk3326.add_node(9, block=io_2)
    add_edge_m2m(bdg=rk3326, src_nodes=[6], dst_blocks=[cortex_a35, mali_g31, vcodec, isp_0, io_2], vol_sum=25.6)

    rk3328 = nx.DiGraph(name="rk3328")
    rk3328.add_nodes_from(range(0, 4), block=cortex_a53)
    rk3328.add_nodes_from(range(4, 6), block=mali_450)
    rk3328.add_node(6, block=ddr_ctrl)
    rk3328.add_node(7, block=vcodec)
    rk3328.add_node(8, block=io_2)
    add_edge_m2m(bdg=rk3328, src_nodes=[6], dst_blocks=[cortex_a53, mali_450, vcodec, io_2], vol_sum=25.6)

    rk3308 = nx.DiGraph(name="rk3308")
    rk3308.add_nodes_from(range(0, 4), block=cortex_a35)
    rk3308.add_node(4, block=ddr_ctrl)
    rk3308.add_node(5, block=io_2)
    add_edge_m2m(bdg=rk3308, src_nodes=[4], dst_blocks=[cortex_a35, io_2], vol_sum=25.6)

    bdg_all = [rk3568, rk3568j, rk3566, rk3399, rk3399pro, rk3288, rk3368, rk3326, rk3328, rk3308]
    assert check_bdgs(bdgs=bdg_all)
    return bdg_all


def Allwinner():
    cortex_a53 = Block(name="cortex_a53", area=8.61, node=28)
    cortex_a7 = Block(name="cortex_a7", area=0.48, node=28)

    mali_g31 = Block(name="mali_g31", area=15.87, node=28)
    powervr_ge8300 = Block(name="powervr_ge8300", area=2.39, node=28)
    mali_t760 = Block(name="mali_t760", area=7.66, node=28)
    powervr_sgx544 = Block(name="powervr_sgx544", area=3.86, node=28)
    mali_t720 = Block(name="mali_t720", area=7.66, node=28)
    mali_400 = Block(name="mali_400", area=7.66, node=28)

    ddr_ctrl = Block(name="ddr_ctrl", area=11.7, node=28)
    vcodec = Block(name="vcodec", area=7.14, node=28)
    isp = Block(name="isp", area=12.54, node=28)
    wifi = Block(name="wifi", area=3.94, node=28)
    audio = Block(name="audio", area=0.10, node=28)
    security = Block(name="security", area=5, node=28)  # unkown
    io = Block(name="io", area=36.53, node=28)

    assert check_blocks(blks=[
        cortex_a53, cortex_a7, mali_g31, powervr_ge8300, mali_t760, powervr_sgx544, mali_t720, mali_400, ddr_ctrl, vcodec,
        audio, security, io
    ])

    t5 = nx.DiGraph()
    t5.add_nodes_from(range(0, 4), block=cortex_a53)
    t5.add_nodes_from(range(4, 6), block=mali_g31)
    t5.add_node(6, block=ddr_ctrl)
    t5.add_node(7, block=vcodec)
    t5.add_node(8, block=audio)
    t5.add_node(9, block=security)
    t5.add_node(10, block=io)

    a133 = nx.DiGraph()
    a133.add_nodes_from(range(0, 4), block=cortex_a53)
    a133.add_node(4, block=powervr_ge8300)
    a133.add_node(5, block=ddr_ctrl)
    a133.add_node(6, block=vcodec)
    a133.add_node(7, block=isp)
    a133.add_node(8, block=audio)
    a133.add_node(9, block=wifi)
    a133.add_node(10, block=io)

    r818 = nx.DiGraph()
    r818.add_nodes_from(range(0, 4), block=cortex_a53)
    r818.add_node(4, block=powervr_ge8300)
    r818.add_node(5, block=ddr_ctrl)
    r818.add_node(6, block=audio)
    r818.add_node(7, block=vcodec)
    r818.add_node(8, block=isp)
    r818.add_node(9, block=security)
    r818.add_node(10, block=io)

    a63 = nx.DiGraph()
    a63.add_nodes_from(range(0, 4), block=cortex_a53)
    a63.add_node(4, block=mali_t760)
    a63.add_node(5, block=ddr_ctrl)
    a63.add_node(6, block=vcodec)
    a63.add_node(7, block=audio)
    a63.add_node(8, block=wifi)
    a63.add_node(9, block=io)

    r328 = nx.DiGraph()
    r328.add_nodes_from(range(0, 2), block=cortex_a7)
    r328.add_node(2, block=ddr_ctrl)
    r328.add_node(3, block=audio)
    r328.add_node(4, block=security)
    r328.add_node(5, block=wifi)
    r328.add_node(6, block=io)

    h616 = nx.DiGraph()
    h616.add_nodes_from(range(0, 4), block=cortex_a53)
    h616.add_nodes_from(range(4, 6), block=mali_g31)
    h616.add_node(6, block=vcodec)
    h616.add_node(7, block=audio)
    h616.add_node(8, block=ddr_ctrl)
    h616.add_node(9, block=security)
    h616.add_node(10, block=io)

    h80 = nx.DiGraph()
    h80.add_nodes_from(range(0, 8), block=cortex_a7)
    h80.add_node(8, block=powervr_sgx544)
    h80.add_node(9, block=isp)
    h80.add_node(10, block=ddr_ctrl)
    h80.add_node(11, block=vcodec)
    h80.add_node(12, block=io)

    a83t_h = nx.DiGraph()
    a83t_h.add_nodes_from(range(0, 8), block=cortex_a7)
    a83t_h.add_node(8, block=powervr_sgx544)
    a83t_h.add_node(9, block=ddr_ctrl)
    a83t_h.add_node(10, block=isp)
    a83t_h.add_node(11, block=vcodec)
    a83t_h.add_node(12, block=io)

    h6 = nx.DiGraph()
    h6.add_nodes_from(range(0, 4), block=cortex_a53)
    h6.add_node(4, block=mali_t720)
    h6.add_node(5, block=vcodec)
    h6.add_node(6, block=io)
    h6.add_node(7, block=audio)
    h6.add_node(8, block=ddr_ctrl)

    a50 = nx.DiGraph()
    a50.add_nodes_from(range(0, 4), block=cortex_a7)
    a50.add_nodes_from(range(4, 6), block=mali_400)
    a50.add_node(6, block=ddr_ctrl)
    a50.add_node(7, block=vcodec)
    a50.add_node(8, block=isp)
    a50.add_node(9, block=security)
    a50.add_node(10, block=io)

    bdg_all = [t5, a133, r818, a63, r328, h616, h80, a83t_h, h6, a50]
    assert check_bdgs(bdg_all)
    return bdg_all


def HiSilicon():
    cpu_cluster = Block(name="cpu_cluster", area=6, node=7)
    llc_slice = Block(name="llc_slice", area=6.5, node=7)
    eio_cluster = Block(name="eio_cluster", area=21, node=14)  # 16nm originally
    usb_cluster = Block(name="usb_cluster", area=21, node=14)
    pcie_cluster = Block(name="pcie_cluster", area=21, node=14)
    his_cluster = Block(name="his_cluster", area=21, node=14)
    int_unit = Block(name="int_unit", area=21, node=14)
    hccs = Block(name="hccs", area=21, node=14)
    b_wireless_acc = Block(name="wireless_acc", area=142, node=14)
    b_davinci_core = Block(name="davinci_core", area=14.25, node=7)
    b_nic_io = Block(name="nic_io", area=142, node=14)

    assert check_blocks(blks=[
        cpu_cluster, llc_slice, eio_cluster, usb_cluster, pcie_cluster, his_cluster, int_unit, hccs, b_wireless_acc,
        b_davinci_core, b_nic_io
    ])

    kunpeng920_server = nx.DiGraph(name="kunpeng920_server")
    kunpeng920_server.add_nodes_from(range(0, 16), block=cpu_cluster)
    kunpeng920_server.add_nodes_from(range(16, 32), block=llc_slice)
    kunpeng920_server.add_node(32, block=eio_cluster)
    kunpeng920_server.add_node(33, block=usb_cluster)
    kunpeng920_server.add_node(34, block=pcie_cluster)
    kunpeng920_server.add_node(35, block=his_cluster)
    kunpeng920_server.add_node(36, block=int_unit)
    kunpeng920_server.add_node(37, block=hccs)
    add_edge_c2c(bdg=kunpeng920_server, cores=list(range(0, 16)), caches=list(range(16, 32)), comm=64)
    add_edge_b2b(bdg=kunpeng920_server, block=llc_slice, vol_sum=64)
    add_edge_m2m(bdg=kunpeng920_server, src_nodes=[34], dst_blocks=[llc_slice], vol_sum=80)

    kunpeng920_lite = nx.DiGraph(name="kunpeng920_lite")
    kunpeng920_lite.add_nodes_from(range(0, 8), block=cpu_cluster)
    kunpeng920_lite.add_nodes_from(range(8, 16), block=llc_slice)
    kunpeng920_lite.add_node(16, block=eio_cluster)
    kunpeng920_lite.add_node(17, block=usb_cluster)
    kunpeng920_lite.add_node(18, block=pcie_cluster)
    kunpeng920_lite.add_node(19, block=his_cluster)
    kunpeng920_lite.add_node(20, block=int_unit)
    kunpeng920_lite.add_node(21, block=hccs)
    add_edge_c2c(bdg=kunpeng920_lite, cores=list(range(0, 8)), caches=list(range(8, 16)), comm=64)
    add_edge_b2b(bdg=kunpeng920_lite, block=llc_slice, vol_sum=64)
    add_edge_m2m(bdg=kunpeng920_lite, src_nodes=[18], dst_blocks=[llc_slice], vol_sum=80)

    pcie_switch = nx.DiGraph(name="pcie_switch")
    pcie_switch.add_node(0, block=eio_cluster)
    pcie_switch.add_node(1, block=usb_cluster)
    pcie_switch.add_node(2, block=pcie_cluster)
    pcie_switch.add_node(3, block=his_cluster)
    pcie_switch.add_node(4, block=int_unit)
    pcie_switch.add_node(5, block=hccs)
    add_edge_m2m(bdg=pcie_switch, src_nodes=[2], dst_blocks=[eio_cluster, usb_cluster, his_cluster, int_unit, hccs], vol_sum=80)

    tiangang = nx.DiGraph(name="tiangang")
    tiangang.add_nodes_from(range(0, 8), block=cpu_cluster)
    tiangang.add_nodes_from(range(8, 16), block=llc_slice)
    tiangang.add_node(16, block=b_wireless_acc)
    add_edge_c2c(bdg=tiangang, cores=list(range(0, 8)), caches=list(range(8, 16)), comm=64)
    add_edge_b2b(bdg=tiangang, block=llc_slice, vol_sum=64)

    ascend_910 = nx.DiGraph(name="ascend_910")
    ascend_910.add_nodes_from(range(0, 32), block=b_davinci_core)
    ascend_910.add_node(32, block=eio_cluster)
    ascend_910.add_node(33, block=usb_cluster)
    ascend_910.add_node(34, block=pcie_cluster)
    ascend_910.add_node(35, block=his_cluster)
    ascend_910.add_node(36, block=int_unit)
    ascend_910.add_node(37, block=hccs)
    add_edge_b2b(bdg=ascend_910, block=b_davinci_core, vol_sum=64)
    add_edge_m2m(bdg=ascend_910, src_nodes=[34], dst_blocks=[b_davinci_core], vol_sum=80)

    hi1822 = nx.DiGraph(name="hi1822")
    hi1822.add_nodes_from(range(0, 8), block=cpu_cluster)
    hi1822.add_nodes_from(range(8, 16), block=llc_slice)
    hi1822.add_node(16, block=b_nic_io)
    add_edge_c2c(bdg=hi1822, cores=list(range(0, 8)), caches=list(range(8, 16)), comm=64)
    add_edge_b2b(bdg=hi1822, block=cpu_cluster, vol_sum=64)

    bdg_all = [kunpeng920_server, kunpeng920_lite, pcie_switch, tiangang, ascend_910, hi1822]
    assert check_bdgs(bdgs=bdg_all)
    return bdg_all


def Nvidia():
    sm = Block(name="sm", area=3.48, node=7)
    L2 = Block(name="L2", area=7.55, node=7)  # 1 MB
    pcie = Block(name="pcie", area=5.18, node=7)  # x16
    hbm_32 = Block(name="hbm_32", area=7.45, node=7)  # 32bit: controller + phy
    hbm_1024_phy= Block(name="hbm_1024_phy", area=9.89, node=7)
    hbm_1536_ctrl = Block(name="hbm_1536_ctrl", area=8.12, node=7)

    ga100 = nx.DiGraph()
    ga100.add_nodes_from(range(0, 128), block=sm)
    ga100.add_nodes_from(range(128, 168), block=L2)  # 40 MB
    ga100.add_node(168, block=pcie)
    ga100.add_nodes_from(range(169, 175), block=hbm_1024_phy)
    ga100.add_nodes_from(range(175, 179), block=hbm_1536_ctrl)
    add_edge_m2m(bdg=ga100, src_nodes=[168], dst_blocks=[L2], vol_sum=64)
    add_edge_m2m(bdg=ga100, src_nodes=range(175, 179), dst_blocks=[L2], vol_sum=774)

    ga102 = nx.DiGraph()
    ga102.add_nodes_from(range(0, 84), block=sm)
    ga102.add_nodes_from(range(84, 90), block=L2)  # 6 MB
    ga102.add_node(90, block=pcie)
    ga102.add_nodes_from(range(91, 103), block=hbm_32)  # 384 bit
    add_edge_m2m(bdg=ga102, src_nodes=[90], dst_blocks=[L2], vol_sum=64)
    add_edge_m2m(bdg=ga102, src_nodes=range(91, 103), dst_blocks=[L2], vol_sum=64)

    ga103 = nx.DiGraph()
    ga103.add_nodes_from(range(0, 60), block=sm)
    ga103.add_nodes_from(range(60, 65), block=L2)  # 5 MB
    ga103.add_node(65, block=pcie)
    ga103.add_nodes_from(range(66, 74), block=hbm_32)  # 256 bit
    add_edge_m2m(bdg=ga103, src_nodes=[65], dst_blocks=[L2], vol_sum=64)
    add_edge_m2m(bdg=ga103, src_nodes=range(66, 74), dst_blocks=[L2], vol_sum=64)

    ga104 = nx.DiGraph()
    ga104.add_nodes_from(range(0, 48), block=sm)
    ga104.add_nodes_from(range(48, 52), block=L2)  # 4 MB
    ga104.add_node(52, block=pcie)
    ga104.add_nodes_from(range(53, 61), block=hbm_32)  # 256 bit
    add_edge_m2m(bdg=ga104, src_nodes=[52], dst_blocks=[L2], vol_sum=64)
    add_edge_m2m(bdg=ga104, src_nodes=range(53, 61), dst_blocks=[L2], vol_sum=64)

    ga106 = nx.DiGraph()
    ga106.add_nodes_from(range(0, 30), block=sm)
    ga106.add_nodes_from(range(30, 33), block=L2)  # 3MB
    ga106.add_node(33, block=pcie)
    ga106.add_nodes_from(range(34, 40), block=hbm_32)  # 192 bit
    add_edge_m2m(bdg=ga106, src_nodes=[33], dst_blocks=[L2], vol_sum=64)
    add_edge_m2m(bdg=ga106, src_nodes=range(33, 39), dst_blocks=[L2], vol_sum=64)

    ga107 = nx.DiGraph()
    ga107.add_nodes_from(range(0, 24), block=sm)
    ga107.add_nodes_from(range(24, 26), block=L2)  # 2MB
    ga107.add_node(26, block=pcie)
    ga107.add_nodes_from(range(27, 31), block=hbm_32)  # 128 bit
    add_edge_m2m(bdg=ga107, src_nodes=[26], dst_blocks=[L2], vol_sum=64)
    add_edge_m2m(bdg=ga107, src_nodes=range(27, 31), dst_blocks=[L2], vol_sum=64)
    

    bdg_all = [ga100, ga102, ga103, ga104, ga106, ga107]
    assert check_bdgs(bdgs=bdg_all)
    return bdg_all


def get_dataset():
    return {"AMD": AMD(), "HiSilicon": HiSilicon(), "Intel": Intel(), "Rockchip": Rockchip(), "Nvidia":Nvidia()}


def Newform():
    cpu_core = Block(name="cpu_core", area=6, node=7)
    fpga_core = Block(name="fpga_core", area=6.5, node=7)

    soc = nx.DiGraph()
    soc.add_nodes_from(range(0, 32), block=cpu_core)
    soc.add_nodes_from(range(32, 64), block=fpga_core)
    return [soc]