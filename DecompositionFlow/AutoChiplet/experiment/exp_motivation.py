import math
from ..model import Block, Chiplet, get_cost, make_package
from .. import utils
from os import path as osp
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib


def plot_2():
    b_cpu = Block(name="cpu", area=20, node=7)
    num_block_1 = 20
    num_block_2 = 24
    type_pkg = "SI"
    cn_1 = sorted(utils.find_factors(num_block_1))
    cn_2 = sorted(utils.find_factors(num_block_2))
    X, Y = np.meshgrid(cn_1, cn_2)
    Z = np.zeros(shape=X.shape)
    vol = 500 * 20
    for i in range(len(X)):
        for j in range(len(X[i])):
            cpl_1 = Chiplet(blocks={b_cpu: num_block_1 // X[i, j]})
            pkg_1 = make_package(type_pkg=type_pkg, chiplets={cpl_1: int(X[i, j])})
            cpl_2 = Chiplet(blocks={b_cpu: num_block_2 // Y[i, j]})
            pkg_2 = make_package(type_pkg=type_pkg, chiplets={cpl_2: int(Y[i, j])})
            cost = get_cost(pkgs=[pkg_1, pkg_2], vols=[vol * 1000, vol * 1000])
            Z[i, j] = np.average(cost)
    print(X)
    print(Y)
    print(Z)
    fig = plt.figure()
    ax = Axes3D(fig=fig, computed_zorder=False)
    # ax = plt.axes(projection='3d', computed_zorder=False)
    ax.view_init(elev=22, azim=129)
    Z_plot = Z / Z.min()
    print(Z_plot)
    ax.plot_surface(X, Y, Z_plot, rstride=1, cstride=1, cmap='summer', linewidth=1, edgecolors='k', alpha=1)
    for i in range(len(X)):
        for j in range(len(X[i])):
            if num_block_1 / X[i, j] != num_block_2 / Y[i, j]:
                Z_plot[i, j] = math.nan
    s_1 = ax.scatter(X, Y, Z_plot, marker="o", s=64, color="b", alpha=1)

    ind = np.unravel_index(Z.argmin(), Z.shape)
    for i in range(len(X)):
        for j in range(len(X[i])):
            if (i, j) != ind:
                Z_plot[i, j] = math.nan
            else:
                Z_plot[i, j] = 0.97
    s_2 = ax.scatter(X, Y, Z_plot, marker="*", s=128, color="r", alpha=1)

    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_xlabel("# chiplets in A", fontname="Arial", fontsize=18)
    ax.set_ylabel("# chiplets in B", fontname="Arial", fontsize=18)
    ax.set_zlabel("Normalized Cost", fontname="Arial", fontsize=18)
    ax.xaxis.set_ticks(np.arange(0, num_block_1 + 1, 5))

    # axbox = ax.get_position()
    # plt.legend((s_1, s_2), ("reuse point", "optimum partition"),
    #            scatterpoints=1,
    #            loc=(axbox.x1, axbox.y0),
    #            fontsize=12)
    fig.tight_layout()
    plt.savefig("motivation.pdf")
    plt.show()

    print("chiplet 1 has {} cores, chiplet 2 has {} cores".format(num_block_1 // X[ind], num_block_2 // Y[ind]))


def get_data():
    b_cpu = Block(name="cpu", area=5, node=7)
    num_block_1 = 20
    num_block_2 = 24
    type_pkg = "SI"
    cn_1 = sorted(utils.find_factors(num_block_1))
    cn_2 = sorted(utils.find_factors(num_block_2))
    X, Y = np.meshgrid(cn_1, cn_2)
    Zs = []
    for vol in [500 * 1000, 10 * 1000 * 1000]:
        Z = np.zeros(shape=X.shape)
        for i in range(len(X)):
            for j in range(len(X[i])):
                cpl_1 = Chiplet(blocks={b_cpu: num_block_1 // X[i, j]})
                pkg_1 = make_package(type_pkg=type_pkg, chiplets={cpl_1: int(X[i, j])})
                cpl_2 = Chiplet(blocks={b_cpu: num_block_2 // Y[i, j]})
                pkg_2 = make_package(type_pkg=type_pkg, chiplets={cpl_2: int(Y[i, j])})
                cost = get_cost(pkgs=[pkg_1, pkg_2], vols=[vol, vol])
                Z[i, j] = np.average(cost)
        Z /= Z.min()
        Z_surface = Z

        Z_dot = Z.copy()
        for i in range(len(X)):
            for j in range(len(X[i])):
                if not math.isclose(num_block_1 / X[i, j], num_block_2 / Y[i, j]):
                    Z_dot[i, j] = math.nan

        ind = np.unravel_index(Z.argmin(), Z.shape)
        Z_star = Z.copy()
        # print("chiplet 1 has {} cores, chiplet 2 has {} cores".format(num_block_1 // X[ind], num_block_2 // Y[ind]))
        for i in range(len(X)):
            for j in range(len(X[i])):
                if (i, j) != ind:
                    Z_star[i, j] = math.nan
        Zs.append((Z_surface, Z_dot, Z_star))
    return X, Y, Zs


def plot():
    X, Y, ((Z_surface_1, Z_dot_1, Z_star_1), (Z_surface_2, Z_dot_2, Z_star_2)) = get_data()
    print(X)
    print(Y)
    print(Z_surface_1)
    print(Z_surface_2)

    label_size = 14
    matplotlib.rcParams["xtick.labelsize"] = label_size
    matplotlib.rcParams["ytick.labelsize"] = label_size
    matplotlib.rcParams["axes.labelsize"] = 14
    matplotlib.rcParams["font.sans-serif"] = "Arial"
    matplotlib.rcParams["xtick.major.pad"] = 0
    matplotlib.rcParams["ytick.major.pad"] = 0

    fig = plt.figure(figsize=plt.figaspect(0.5))
    plt.subplots_adjust(wspace=0.15)

    ax_1 = fig.add_subplot(1, 2, 1, projection='3d', computed_zorder=False)
    ax_1.set_title("(a) volume of 500k per design", y=0.92, fontsize=18, weight="bold")
    ax_1.view_init(elev=18, azim=157)
    ax_1.plot_surface(X, Y, Z_surface_1, rstride=1, cstride=1, cmap='summer', linewidth=1, edgecolors='k', alpha=1)
    ax_1.scatter(X, Y, Z_dot_1, marker="o", s=64, color="b", alpha=1)
    ax_1.scatter(X, Y, Z_star_1, marker="*", s=128, color="r", alpha=1)
    ax_1.set_xlabel("# chiplets of design-A", weight="bold")
    ax_1.set_ylabel("# chiplets of design-B", weight="bold")
    ax_1.set_zlabel("Normalized Cost", weight="bold")
    ax_1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax_2 = fig.add_subplot(1, 2, 2, projection='3d', computed_zorder=False)
    ax_2.set_title("(b) volume of 10M per design", y=0.92, fontsize=18, weight="bold")
    ax_2.view_init(elev=18, azim=157)
    ax_2.plot_surface(X, Y, Z_surface_2, rstride=1, cstride=1, cmap='summer', linewidth=1, edgecolors='k', alpha=1)
    dots = ax_2.scatter(X, Y, Z_dot_2, marker="o", s=64, color="b", alpha=1)
    stars = ax_2.scatter(X, Y, Z_star_2, marker="*", s=128, color="r", alpha=1)
    ax_2.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_2.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_2.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_2.zaxis.set_ticks(np.arange(1, Z_surface_2.max() + 0.2, 0.4))

    fig.legend([dots, stars], ["Reusable Partition", "Optimal Partition"], loc='lower center', ncol=2, fontsize=12)
    plt.savefig("motivation.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot()