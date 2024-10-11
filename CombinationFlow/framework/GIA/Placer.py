import math
import pickle
from rectpack import newPacker
import numpy as np
from ..ChipletSys import ChipletSys, is_overlap_with_rects
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


class PlacerSA:
    """
        Chiplet placement using 2-stage simulated annealing.
        Use class to save state more easily.
    """
    def __init__(self, csys, cost_func, dir_log, cfg_algo=None, pid=0):
        """
            csys: the ChipletSys need to place chiplets
            cost_func: function, for two-stage algo, no need to generate Mlayout at thermal optimization
                parameter: (csys, placement, stage, nv_dict), stage: "init_therm", "therm_opt", "init_joint", "joint_opt"; nv_dict: normalized value dict
                Return: (cost_value, cost_items), cost_value is a single float and cost_items dict(power, perf, thermal, cost, mlayout)
            dir_log: dir to output curve of opitmization
            pid: can be used to reseed for parallel SA
            dir_log: dir to output log, e.g. figures
        """
        if not type(csys) == ChipletSys:
            raise TypeError("ERROR: csys should be ChipletSys instance")

        np.random.seed(seed=pid)  # this is important for multiprocess SA
        self.pid = pid

        self.csys = csys
        self.cost_func = cost_func
        self.dir_log = dir_log
        self.cur_pl = None  # current accepted placement. initialized in init_sample()
        self.cur_cv = None  # cost value of current accepted solution
        self.best_sol = None  # the best solution in history, format: (placement, cost_value, cost_items)

        if cfg_algo is None:
            cfg_algo = {}
        self.max_try = cfg_algo["max_try"] if "max_try" in cfg_algo else 20  # number of retry for single perturb action
        self.T_start_t = cfg_algo["T_start_t"] if "T_start_t" in cfg_algo else 100  # initial temperature of thermal optimization
        self.T_end_t = cfg_algo["T_end_t"] if "T_end_t" in cfg_algo else 0.05
        self.T_start_j = cfg_algo["T_start_j"] if "T_start_j" in cfg_algo else 50  # initial temperature of joint optimization
        self.T_end_j = cfg_algo["T_end_j"] if "T_end_j" in cfg_algo else 0.1
        self.init_n_t = cfg_algo["init_n_t"] if "init_n_t" in cfg_algo else 10  # number of sampling before thermal optimization
        self.init_n_j = cfg_algo["init_n_j"] if "init_n_j" in cfg_algo else 20  # number of sampling before joint optimization
        self.T_topt = cfg_algo[
            "T_topt"] if "T_topt" in cfg_algo else 85  # the target of temperature optimization, joint optimization begins when reaching it
        self.weight_act_topt = cfg_algo["weight_act_topt"] if "weight_act_topt" in cfg_algo else [
            0.4, 0, 0.4, 0.2
        ]  # action distribution of MOVE/ROTATE/JUMP/SWAP in thermal optimization
        self.max_move_topt = cfg_algo[
            "max_move_topt"] if "max_move_topt" in cfg_algo else 3  # the max step of moving chiplets in thermal optimization
        self.weight_act_jopt = cfg_algo["weight_act_jopt"] if "weight_act_jopt" in cfg_algo else [0.25, 0.25, 0.25, 0.25
                                                                                                  ]  # joint optimization
        self.max_move_jopt = cfg_algo["max_move_jopt"] if "max_move_jopt" in cfg_algo else 1

    def neighbor(self, placement, weight_act, max_move):
        """
            Perturb the placement.
            Return: None if failed else new placement(similar to deepcopy)
            
            csys: ChipletSys instance
            placement: the placement to be changed/perturbed, will not be changed
        """
        MOVE = 0  # randomly move a chiplet by one tile in up, down, left or right.
        ROTATE = 1  # randomly rotate a chiplet
        JUMP = 2  # randomly set a chiplet to a valid position
        SWAP = 3  # randomly swap two chiplet
        for act_try in range(self.max_try):
            act = np.random.choice([MOVE, ROTATE, JUMP, SWAP], size=1, p=weight_act)[0]
            if act == MOVE:
                for _ in range(self.max_try):
                    target = np.random.randint(low=0, high=len(self.csys.chiplets))  # the selected chiplet
                    direction = np.random.randint(low=0, high=4)
                    rects = [(cx, cy, cx + cpl.w(angle), cy + cpl.h(angle))
                             for idx_c, ((cx, cy, angle), cpl) in enumerate(zip(placement, self.csys.chiplets))
                             if idx_c != target]
                    cx_t, cy_t, angle_t = placement[target]
                    w_t = self.csys.chiplets[target].w(angle_t)
                    h_t = self.csys.chiplets[target].h(angle_t)

                    tile_move = np.random.randint(low=1, high=max_move + 1)  # the moved
                    if direction == 0:  # up
                        cy_t += tile_move
                    elif direction == 1:  # down
                        cy_t -= tile_move
                    elif direction == 2:  # left
                        cx_t -= tile_move
                    else:  # right
                        cx_t += tile_move

                    if cx_t < 0 or cy_t < 0 or (cx_t + w_t) > self.csys.W or (cy_t + h_t) > self.csys.H:  # out of interposer
                        continue
                    if is_overlap_with_rects(x_bl=cx_t, y_bl=cy_t, x_ur=cx_t + w_t, y_ur=cy_t + h_t, rects=rects):
                        continue
                    new_placement = [(cx, cy, angle) if idx_c != target else (cx_t, cy_t, angle_t)
                                     for idx_c, (cx, cy, angle) in enumerate(placement)]
                    # print("MOVE: from {} to {}".format(placement[target], new_placement[target]))
                    return new_placement

            elif act == ROTATE:
                for _ in range(self.max_try):
                    target = np.random.randint(low=0, high=len(self.csys.chiplets))  # the selected chiplet
                    rects = [(cx, cy, cx + cpl.w(angle), cy + cpl.h(angle))
                             for idx_c, ((cx, cy, angle), cpl) in enumerate(zip(placement, self.csys.chiplets))
                             if idx_c != target]
                    cx_t, cy_t, angle_t = placement[target]

                    assert angle_t == 0 or angle_t == 1
                    angle_t = 0 if angle_t == 1 else 1
                    w_t = self.csys.chiplets[target].w(angle_t)
                    h_t = self.csys.chiplets[target].h(angle_t)

                    assert cx_t >= 0 or cy_t >= 0
                    if (cx_t + w_t) > self.csys.W or (cy_t + h_t) > self.csys.H:  # out of interposer
                        continue
                    if is_overlap_with_rects(x_bl=cx_t, y_bl=cy_t, x_ur=cx_t + w_t, y_ur=cy_t + h_t, rects=rects):
                        continue

                    new_placement = [(cx, cy, angle) if idx_c != target else (cx_t, cy_t, angle_t)
                                     for idx_c, (cx, cy, angle) in enumerate(placement)]
                    # print("ROTATE: from {} to {}".format(placement[target], new_placement[target]))
                    return new_placement

            elif act == JUMP:
                for _ in range(self.max_try):
                    target = np.random.randint(low=0, high=len(self.csys.chiplets))  # the selected chiplet
                    angle_t = placement[target][2]
                    rects = [(cx, cy, cx + cpl.w(angle), cy + cpl.h(angle))
                             for idx_c, ((cx, cy, angle), cpl) in enumerate(zip(placement, self.csys.chiplets))
                             if idx_c != target]
                    w_t = self.csys.chiplets[target].w(angle_t)
                    h_t = self.csys.chiplets[target].h(angle_t)

                    valid_pos = []
                    for x_intp in range(self.csys.W):
                        for y_intp in range(self.csys.H):
                            if (x_intp + w_t) > self.csys.W or (y_intp + h_t) > self.csys.H:
                                continue
                            if is_overlap_with_rects(x_bl=x_intp,
                                                     y_bl=y_intp,
                                                     x_ur=x_intp + w_t,
                                                     y_ur=y_intp + h_t,
                                                     rects=rects):
                                continue
                            valid_pos.append((x_intp, y_intp))

                    if not valid_pos:  # no valid position
                        continue
                    idx_pos = np.random.randint(0, len(valid_pos))
                    cx_t, cy_t = valid_pos[idx_pos]
                    new_placement = [(cx, cy, angle) if idx_c != target else (cx_t, cy_t, angle_t)
                                     for idx_c, (cx, cy, angle) in enumerate(placement)]
                    # print("JUMP: from {} to {}".format(placement[target], new_placement[target]))
                    return new_placement

            elif act == SWAP:
                for _ in range(self.max_try):
                    target_1, target_2 = np.random.choice(list(range(len(self.csys.chiplets))), size=2, replace=False)

                    cx_t1, cy_t1, angle_t1 = placement[target_1]
                    w_t1 = self.csys.chiplets[target_1].w(angle_t1)
                    h_t1 = self.csys.chiplets[target_1].h(angle_t1)

                    cx_t2, cy_t2, angle_t2 = placement[target_2]
                    w_t2 = self.csys.chiplets[target_2].w(angle_t2)
                    h_t2 = self.csys.chiplets[target_2].h(angle_t2)

                    # boundary check
                    if cx_t2 + w_t1 > self.csys.W or cx_t1 + w_t2 > self.csys.W or cy_t2 + h_t1 > self.csys.H or cy_t1 + h_t2 > self.csys.H:
                        continue

                    rects = [(cx, cy, cx + cpl.w(angle), cy + cpl.h(angle))
                             for idx_c, ((cx, cy, angle), cpl) in enumerate(zip(placement, self.csys.chiplets))
                             if idx_c != target_1 and idx_c != target_2]

                    rect_t1 = (cx_t2, cy_t2, cx_t2 + w_t1, cy_t2 + h_t1)  # set cx/cy of target1 to target2
                    if not is_overlap_with_rects(*rect_t1, rects=rects):
                        rects.append(rect_t1)
                        rect_t2 = (cx_t1, cy_t1, cx_t1 + w_t2, cy_t1 + h_t2)  # set cx/cy of target2 to target1

                        if not is_overlap_with_rects(*rect_t2, rects=rects):
                            # print("SWAP: {} and {}".format(placement[target_1], placement[target_2]))
                            new_placement = [(cx, cy, angle) for (cx, cy, angle) in placement]
                            new_placement[target_1] = (cx_t2, cy_t2, angle_t1)
                            new_placement[target_2] = (cx_t1, cy_t1, angle_t2)
                            return new_placement

            else:
                assert False

        return None  # failed to generate new placement

    def init_therm(self):
        """
            1. Use rectpack to generate inital compact placement
            2. Randomly perturb the placement before thermal optimization
        """
        packer = newPacker()
        for idx_cpl, cpl in enumerate(self.csys.chiplets):
            packer.add_rect(width=cpl.w_orig, height=cpl.h_orig, rid=idx_cpl)
        packer.add_bin(width=self.csys.W, height=self.csys.H, count=1)
        packer.pack()
        all_rects = packer.rect_list()
        assert len(all_rects) == len(self.csys.chiplets)
        init_pl = [0] * len(self.csys.chiplets)  # initial placement
        for rect in all_rects:
            _, x, y, w, h, rid = rect
            assert (w == self.csys.chiplets[rid].w_orig
                    and h == self.csys.chiplets[rid].h_orig) or (h == self.csys.chiplets[rid].w_orig
                                                                 and w == self.csys.chiplets[rid].h_orig)
            angle = 0 if (w == self.csys.chiplets[rid].w_orig and h == self.csys.chiplets[rid].h_orig) else 1
            init_pl[rid] = (x, y, angle)

        self.cur_pl = init_pl
        init_cvi = self.cost_func(self.csys, init_pl, "init_therm", nv_dict=None)

        p = init_pl
        it_T = []  # thermal of initial sampling
        for _ in range(self.init_n_t):
            new_p = self.neighbor(placement=p, weight_act=self.weight_act_topt, max_move=self.max_move_topt)
            if new_p is not None:
                p = new_p
                it_T.append(self.cost_func(self.csys, new_p, "init_therm", nv_dict=None)[1]["thermal"])
        self.nv_dict = {"thermal": np.average(it_T)}  # normalized cost value of init thermal
        self.cur_cv = init_cvi[1]["thermal"] / self.nv_dict["thermal"]  # Normalized
        self.cur_ci = init_cvi[1]
        self.best_sol = (init_pl, self.cur_cv, self.cur_ci)
        self.it_sol = (init_pl, self.cur_cv, self.cur_ci)  # solution of init thermal optimization

    def topt(self):  # thermal optimization
        """
            Optimize until reaching T_topt or max_step_topt
        """
        mpl.use('agg')
        self.init_therm()  # random sample before thermal optimization
        T = self.T_start_t  # set annealing temperature
        hist = {"cv": [], "thermal": []}  # history values
        idx_step = 0
        while True:
            neigh_pl = self.neighbor(placement=self.cur_pl, weight_act=self.weight_act_topt, max_move=self.max_move_topt)
            if neigh_pl is None:  # find neighboring solution failed
                continue
            neigh_cvi = self.cost_func(self.csys, neigh_pl, "therm_opt", self.nv_dict)
            neigh_cv = neigh_cvi[0]  # cost value of this loop
            neigh_ci = neigh_cvi[1]
            delta_cv = self.cur_cv - neigh_cv
            AP = min(1, math.exp(delta_cv / T))  # acceptance probability
            if np.random.rand() <= AP:  # accept
                self.cur_cv = neigh_cv
                self.cur_ci = neigh_ci
                self.cur_pl = neigh_pl
                if neigh_cv < self.best_sol[1]:
                    self.best_sol = (neigh_pl, neigh_cv, neigh_ci)  # format: (placement, cost_value, cost_items)

            # log for debug
            hist["cv"].append(self.cur_cv)
            hist["thermal"].append(self.cur_ci["thermal"])
            if idx_step % 1 == 0:  # no need to refresh in every iteration
                _, axes = plt.subplots(1, 2, figsize=(10, 5))
                df = pd.DataFrame(hist)
                sns.lineplot(ax=axes[0], data=df["cv"])
                sns.lineplot(ax=axes[1], data=df["thermal"])
                plt.savefig(os.path.join(self.dir_log, "{}.png".format(self.pid)))
                plt.close()

            T = self.T_start_t / (idx_step + 1)
            if T < self.T_end_t or self.best_sol[2]["thermal"] <= self.T_topt:  # end of SA
                break
            idx_step += 1
        return self.best_sol, self.it_sol

    def init_joint(self, placement):
        """
            Randomly perturb the placement before joint optimization
        """
        self.cur_pl = placement
        init_cvi = self.cost_func(self.csys, placement, "init_joint", nv_dict=None)

        p = placement
        ij_cvi = []  # cost items of initial joint sampling
        for _ in range(self.init_n_j):
            new_p = self.neighbor(placement=p, weight_act=self.weight_act_jopt, max_move=self.max_move_jopt)
            if new_p is not None:
                p = new_p
                ij_cvi.append(self.cost_func(self.csys, new_p, "init_joint", nv_dict=None)[1])
        keys = ["power_eu", "perf", "thermal", "pcost"]  # for active interposer, "pcost" should be 0
        nv_dict_ = dict([(k, []) for k in keys])
        for k in keys:
            for item in ij_cvi:
                nv_dict_[k].append(item[k])

        self.nv_dict = dict([(k, np.average(v)) for k, v in nv_dict_.items()])
        self.cur_cv = init_cvi[0]
        self.cur_ci = init_cvi[1]
        self.best_sol = (placement, self.cur_cv, self.cur_ci)
        self.ij_sol = (placement, self.cur_cv, self.cur_ci)  # solution of init joint optimization

    def jopt(self, placement):
        """
            joint optimization
            placement: initial placement of joint optimization
        """
        mpl.use('agg')
        self.init_joint(placement=placement)  # random sample before joint optimization
        T = self.T_start_j  # set annealing temperature
        hist = {"cv": [], "power_eu": [], "perf": [], "thermal": [], "pcost": []}  # history cost values
        idx_step = 0
        while True:
            neigh_pl = self.neighbor(placement=self.cur_pl, weight_act=self.weight_act_jopt, max_move=self.max_move_jopt)
            if neigh_pl is None:  # find neighboring solution failed
                continue
            neigh_cvi = self.cost_func(self.csys, neigh_pl, "joint_opt", self.nv_dict)
            neigh_cv = neigh_cvi[0]  # cost value of this loop
            neigh_ci = neigh_cvi[1]
            delta_cv = self.cur_cv - neigh_cv
            AP = min(1, math.exp(delta_cv / T))  # acceptance probability
            if np.random.rand() <= AP:  # accept
                self.cur_cv = neigh_cv
                self.cur_ci = neigh_ci
                self.cur_pl = neigh_pl
                if neigh_cv < self.best_sol[1]:
                    self.best_sol = (neigh_pl, neigh_cv, neigh_ci)  # format: (placement, cost_value, cost_items)
                    with open(os.path.join(self.dir_log, "{}.pkl".format(self.pid)), "wb") as out_p:
                        pickle.dump(self.best_sol, out_p)

            # log for debug
            hist["cv"].append(self.cur_cv)
            hist["power_eu"].append(self.cur_ci["power_eu"])
            hist["thermal"].append(self.cur_ci["thermal"])
            hist["perf"].append(self.cur_ci["perf"])
            hist["pcost"].append(self.cur_ci["pcost"])

            if idx_step % 2 == 0:  # no need to refresh in every iteration
                _, axes = plt.subplots(3, 2, figsize=(14, 15))
                df = pd.DataFrame(hist)
                sns.lineplot(ax=axes[0, 0], data=df["cv"])
                sns.lineplot(ax=axes[0, 1], data=df["power_eu"])
                sns.lineplot(ax=axes[1, 0], data=df["thermal"])
                sns.lineplot(ax=axes[1, 1], data=df["perf"])
                sns.lineplot(ax=axes[2, 0], data=df["pcost"])
                sns.lineplot(ax=axes[2, 1], data=df)
                plt.savefig(os.path.join(self.dir_log, "{}.png".format(self.pid)))
                plt.close()

            T = self.T_start_j / (idx_step + 1)
            if T < self.T_end_j:  # reach the max step
                with open(os.path.join(self.dir_log, "{}.pkl".format(self.pid)), "wb") as out_p:
                        pickle.dump(self.best_sol, out_p)
                break
            idx_step += 1
        return self.best_sol