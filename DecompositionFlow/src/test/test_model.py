from ..model import *
from matplotlib import pyplot as plt


def test_package_1():
    b_0 = Block("cpu", area=1, node=7)
    b_1 = Block("gpu", area=0.88, node=7)
    assert b_0 == b_0 and hash(b_0) == hash(b_0)
    assert b_1 == b_1 and hash(b_1) == hash(b_1)
    assert b_0 != b_1
    b_2 = Block("cpu", area=1, node=7)
    b_3 = Block("gpu", area=3, node=7)
    assert Chiplet({b_0: 2, b_1: 2}) == Chiplet({b_3: 2, b_2: 2})
    assert Chiplet({b_0: 2, b_0: 2}) != Chiplet({b_3: 2, b_2: 2})
    assert Chiplet({b_0: 2}).area == 2
    pkg = Package(chiplets={Chiplet({b_0: 2, b_1: 2}): 10})
    print(pkg.area_chiplets(), pkg.num_chiplets())


def test_package_2():
    die_yield_3, die_yield_5, die_yield_7 = [], [], []
    for i in range(1, 900):
        b3 = Block(name="cpu", area=i, node="3")
        b5 = Block(name="cpu", area=i, node="5")
        b7 = Block(name="cpu", area=i, node="7")
        cpl_3 = Chiplet({b3: 1}, comm=(20, 20))
        cpl_5 = Chiplet({b5: 1}, comm=(20, 20))
        cpl_7 = Chiplet({b7: 1}, comm=(20, 20))
        die_yield_3.append(cpl_3.die_yield() * 100)
        die_yield_5.append(cpl_5.die_yield() * 100)
        die_yield_7.append(cpl_7.die_yield() * 100)
    plt.plot(die_yield_3, label="3")
    plt.plot(die_yield_5, label="5")
    plt.plot(die_yield_7, label="7")
    plt.legend(loc="upper right")
    plt.ylim(0, 100)
    plt.show()


def test_package_3():
    RE_3, RE_5, RE_7 = [], [], []
    for i in range(1, 900):
        b3 = Block(name="cpu", area=i, node=3)
        b5 = Block(name="cpu", area=i, node=5)
        b7 = Block(name="cpu", area=i, node=7)
        cpl_3 = Chiplet({b3: 1}, comm=(20, 20))
        cpl_5 = Chiplet({b5: 1}, comm=(20, 20))
        cpl_7 = Chiplet({b7: 1}, comm=(20, 20))
        RE_3.append(cpl_3.cost_KGD())
        RE_5.append(cpl_5.cost_KGD())
        RE_7.append(cpl_7.cost_KGD())
    plt.plot(RE_3, label="3")
    plt.plot(RE_5, label="5")
    plt.plot(RE_7, label="7")
    plt.legend(loc="upper right")
    plt.show()


def test_package_4():
    sunny_cove = Block(name="sunny_cove", area=9.04, node=10)
    pcie = Block(name="pcie", area=7.82, node=10)  # 16 lanes
    upi = Block(name="upi", area=18.75, node=10)
    ddr = Block(name="ddr", area=27.88, node=10)  # 2 channels

    cpl_1 = Chiplet(blocks={sunny_cove: 8, pcie: 4}, comm=(0, 0))
    cpl_2 = Chiplet(blocks={upi: 2, ddr: 4}, comm=(0, 0))
    pkg_all = [make_package(type_pkg="SI", chiplets={cpl_1: 2, cpl_2: 1})]
    vol_all = [500 * 1000]
    print(get_cost(pkgs=pkg_all, vols=vol_all))

    cpl_1 = Chiplet(blocks={sunny_cove: 8, pcie: 4}, comm=(370, 370))
    cpl_2 = Chiplet(blocks={upi: 2, ddr: 4}, comm=(370, 370))
    pkg_all = [make_package(type_pkg="SI", chiplets={cpl_1: 2, cpl_2: 1})]
    vol_all = [500 * 1000]
    print(get_cost(pkgs=pkg_all, vols=vol_all))


if __name__ == "__main__":
    test_package_4()