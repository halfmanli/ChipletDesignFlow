from typing import List
from .bstree import Bstree
from copy import deepcopy
import random, math, os
from rectpack import newPacker


def accept_probability(wl_current, wl_new, area_current, area_new, T, step):
    # assume the weights for wirelength and area term are equal
    if wl_min != wl_max and area_min != area_max:
        old_cost = (wl_current - wl_min) / (
            wl_max - wl_min
        )  # 0.5 * (wl_current - wl_min) / (wl_max - wl_min) + 0.5 * (area_current - area_min) / (area_max - area_min)
        new_cost = (wl_new - wl_min) / (
            wl_max - wl_min
        )  # 0.5 * (wl_new - wl_min) / (wl_max - wl_min) + 0.5 * (area_new - area_min) / (area_max - area_min)
    else:
        old_cost = (wl_current - wl_min)  # 0.5 * (wl_current - wl_min) + 0.5 * (area_current - area_min)
        new_cost = (wl_new - wl_min)  # 0.5 * (wl_new - wl_min) + 0.5 * (area_new - area_min)
    global cost_chg_avg
    cost_chg_avg = (cost_chg_avg * (step - 1) + abs(new_cost - old_cost)) / step
    delta = -(new_cost - old_cost)
    if delta > 0:
        ap = 1
    else:
        ap = math.exp(delta / T)
    return ap


def get_connections(connection_matrix):
    # get connection information. One time execution
    s, t = [], []
    net, wire_count = 0, 0
    n_chiplet = len(connection_matrix)
    for i in range(n_chiplet):
        for j in range(n_chiplet):
            if (i != j) and (connection_matrix[i][j] > 0):
                s.append(i)
                t.append(j)
                net += 1
                wire_count += connection_matrix[i][j]
    return net, s, t, wire_count


def compute_wirelength(tree, step, connection_matrix):
    global spacing_
    # length_per_wire value, do not normalize
    total_wirelength = 0
    num_cpl = len(tree.ind_arr)
    wlm = [[0] * num_cpl for _ in range(num_cpl)]
    for i in range(net):
        s_index = tree.ind_arr.index(s[i])
        t_index = tree.ind_arr.index(t[i])
        # wirelength = (abs(tree.x_arr[s_index] + tree.width_arr[s_index] / 2 - tree.x_arr[t_index] - tree.width_arr[t_index] / 2)
        #               + abs(tree.y_arr[s_index] + tree.height_arr[s_index] / 2 - tree.y_arr[t_index] -
        #                     tree.height_arr[t_index] / 2)) * connection_matrix[s_index][t_index]
        x_overlap, y_overlap = False, False
        if tree.x_arr[s_index] >= tree.x_arr[t_index] + tree.width_arr[t_index] or tree.x_arr[
                t_index] >= tree.x_arr[s_index] + tree.width_arr[s_index]:
            dx = abs(tree.x_arr[s_index] + tree.width_arr[s_index] / 2 - tree.x_arr[t_index] -
                     tree.width_arr[t_index] / 2) - (tree.width_arr[s_index] / 2 + tree.width_arr[t_index] / 2 - spacing_)
        else:
            dx = 0
            x_overlap = True

        if tree.y_arr[s_index] >= tree.y_arr[t_index] + tree.height_arr[t_index] or tree.y_arr[
                t_index] >= tree.y_arr[s_index] + tree.height_arr[s_index]:
            dy = abs(tree.y_arr[s_index] + tree.height_arr[s_index] / 2 - tree.y_arr[t_index] -
                     tree.height_arr[t_index] / 2) - (tree.height_arr[s_index] / 2 + tree.height_arr[t_index] / 2 - spacing_)
        else:
            dy = 0
            y_overlap = True

        wirelength = dx + dy
        total_wirelength += wirelength * connection_matrix[s_index][t_index]
        wlm[s_index][t_index] = wirelength
        assert not (x_overlap and y_overlap)
        assert math.isclose(wirelength, spacing_) or wirelength > spacing_

    wl = total_wirelength / (wire_count + 0.0001)
    # update the wirelength stats for normalization
    global wl_max, wl_min
    if wl > wl_max:
        wl_max = wl
    if wl < wl_min:
        wl_min = wl
    return wl, wlm


def compute_area(tree, step):
    n_chiplet = len(tree.ind_arr)
    edge = 0
    for i in range(n_chiplet):
        if edge < tree.x_arr[i] + tree.width_arr[i]:
            edge = tree.x_arr[i] + tree.width_arr[i]
        if edge < tree.y_arr[i] + tree.height_arr[i]:
            edge = tree.y_arr[i] + tree.height_arr[i]
    global area_max, area_min
    if edge > area_max:
        area_max = edge
    if edge < area_min:
        area_min = edge
    return edge


def neighbor(tree):
    tree_new = deepcopy(tree)
    n_chiplet = len(tree.ind_arr)
    op_dice = random.randint(0, n_chiplet + 2 * n_chiplet * n_chiplet + n_chiplet * (n_chiplet - 1) / 2 - 1)
    if op_dice < n_chiplet:
        # rotate, only determine which node to rotate
        tree_new.rotate(tree_new.find_node(tree_new.root, tree_new.ind_arr[op_dice]))
    elif n_chiplet <= op_dice < n_chiplet + n_chiplet * (n_chiplet - 1) / 2:
        # swap, determine two nodes
        n1 = random.randint(0, n_chiplet - 1)
        n2 = random.randint(0, n_chiplet - 1)
        while n2 == n1:
            n2 = random.randint(0, n_chiplet - 1)
        node1 = tree_new.find_node(tree_new.root, tree_new.ind_arr[n1])
        node2 = tree_new.find_node(tree_new.root, tree_new.ind_arr[n2])
        tree_new.swap(node1, node2)
    else:
        # move, determine the node to move, and the target position (left/right child of other nodes or insert to replace root)
        n1 = random.randint(0, n_chiplet - 1)
        n2 = random.randint(0, n_chiplet - 1)
        d = random.randint(0, 1)
        dirs = 'right' if d else 'left'
        node1 = tree_new.find_node(tree_new.root, tree_new.ind_arr[n1])  # the node to be moved
        node2 = tree_new.find_node(tree_new.root,
                                   tree_new.ind_arr[n2])  # the parent node that the moved node is going to insert to.
        if n1 == n2:
            if tree_new.root == node2:
                return neighbor(tree)
            node2 = tree_new.root.parent
        tree_new.move(node1, node2, dirs)
    tree_new.reconstruct()
    return tree_new


def anneal(ind, x, y, width, height, connection_matrix):
    # generate initial placement, and evaluate initial cost
    step, step_best = 1, 1
    tree = Bstree()
    tree.flp2bstree(ind, x, y, width, height)
    tree.reconstruct()
    tree_best = deepcopy(tree)
    global net, s, t, wire_count
    net, s, t, wire_count = get_connections(connection_matrix)
    global wl_max, wl_min, cost_chg_avg
    global area_max, area_min  # we use the longer edge length to represent interposer area
    wl_max, wl_min = 0, 100
    area_max, area_min = 0, 100
    cost_chg_avg = 0
    wl_current, wlm_current = compute_wirelength(tree, step, connection_matrix)
    wl_best = wl_current
    wlm_best = wlm_current
    area_current = compute_area(tree, step)
    area_best = area_current

    T = 1
    T_min = 0.01
    step_size = 1
    alpha = 0.99  # temperature decay factor
    while T > T_min:
        i = 1
        while i <= step_size:
            tree_new = neighbor(tree)
            wl_new, wlm_new = compute_wirelength(tree_new, step, connection_matrix)
            area_new = compute_area(tree_new, step)
            ap = accept_probability(wl_current, wl_new, area_current, area_new, T, step)
            r = random.random()
            if ap > r:
                tree = deepcopy(tree_new)
                wl_current = wl_new
                wlm_current = wlm_new
                area_current = area_new
                if wl_current < wl_best:
                    wl_best = wl_current
                    wlm_best = wlm_current
                    area_best = area_current
                    tree_best = deepcopy(tree)
                    step_best = step
            step += 1
            i += 1
        T *= alpha
    return tree_best, step_best, wl_best, wlm_best, area_best


def place(areas: List[float], connection_matrix: List[List[int]], spacing: float = 0.1):
    assert len(areas) == len(connection_matrix) == len(connection_matrix[0])

    W, H = 40, 40
    n_cpl = len(areas)
    width = list(map(math.sqrt, areas))
    height = width[:]
    width = [w + spacing for w in width]
    height = [h + spacing for h in height]
    ind = list(range(n_cpl))
    x = [-1] * n_cpl
    y = [-1] * n_cpl
    global spacing_
    spacing_ = spacing

    packer = newPacker()
    for idx_cpl, (width_cpl, height_cpl) in enumerate(zip(width, height)):
        packer.add_rect(width=width_cpl, height=height_cpl, rid=idx_cpl)
    packer.add_bin(width=W, height=H, count=1)
    packer.pack()
    all_rects = packer.rect_list()
    assert len(all_rects) == len(areas)

    for rect in all_rects:
        _, x_cpl, y_cpl, _, _, rid_cpl = rect
        x[rid_cpl] = x_cpl
        y[rid_cpl] = y_cpl

    tree_best, step_best, wl_best, wlm_best, _ = anneal(ind, x, y, width, height, connection_matrix)
    # tree_best.printTree(tree_best.root)
    # print('step_best = ', step_best, 'wirelength = ', wl_best)
    return wlm_best


def check_placer():
    global spacing_
    spacing_ = 0.1
    # initial placement
    # node   0  1    2  3    4    5  6  7
    ind = [0, 1, 2, 3, 4, 5, 6, 7]
    x = [0, 3, 0, 3, 5, 2, 0, 3]
    y = [0, 0, 2, 1.5, 1.5, 3, 5, 4]
    width = [3, 4, 2, 2, 1, 4, 3, 4]
    height = [2, 1.5, 3, 1.5, 1, 1, 2, 2]

    # example 2
    ind = [0, 1, 2, 3, 4, 5, 6, 7]
    x = [0, 0, 0, 0, 0, 0, 0, 0]
    y = [0, 0, 0, 0, 0, 0, 0, 0]
    width = [3, 4, 2, 2, 1, 4, 3, 4]
    height = [2, 1.5, 3, 1.5, 1, 1, 2, 2]

    connection_matrix = [[0, 128, 128, 0, 0, 0, 0, 128], [128, 0, 128, 0, 0, 0, 128, 0], [128, 128, 0, 128, 128, 128, 128, 128],
                         [0, 0, 128, 0, 0, 0, 0, 0], [0, 0, 128, 0, 0, 0, 0, 0], [0, 0, 128, 0, 0, 0, 0, 0],
                         [0, 128, 128, 0, 0, 0, 0, 128], [128, 0, 128, 0, 0, 0, 128, 0]]
    n_chiplet = len(connection_matrix)

    tree_best, step_best, wl_best, _ = anneal(ind, x, y, width, height, connection_matrix)
    tree_best.printTree(tree_best.root)
    print('step_best = ', step_best, 'wirelength = ', wl_best)
