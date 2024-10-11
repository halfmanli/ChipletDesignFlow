import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.cm as cm


class Chiplet:
    def __init__(self, name, w, h, power, pins):
        """
            pins: list of pin position, [(px_0, py_1)]
            py
            ^
            |
            |------w-----|
            |            |
            h            |
            |            |
          (0,0)----------| -> px
        """
        self.name = name
        self.__w = w  # unit: tile size; w, h, pins might change due to rotation
        self.__h = h
        self.power = power
        self.__pins = copy.deepcopy(pins)
        self.check_param()

    def check_param(self):
        # w, h check
        if not (type(self.__w) == int and type(self.__h) == int):
            raise TypeError("ERROR: Chiplet width({}) or height({}) is not int".format(type(self.__w), type(self.__h)))
        if not (self.__w > 0 and self.__h > 0):
            raise ValueError("ERROR: Non-positive chiplet width({}) or height({})".format(self.__w, self.__h))

        # pin position check
        if len(self.__pins) == 0:
            raise ValueError("ERROR: empty pins")
        if not all(type(px) == int and type(py) == int for (px, py) in self.__pins):
            raise TypeError("ERROR: px or py is not int, {}".format(self.__pins))
        if not all(px >= 0 and px < self.__w and py >= 0 and py < self.__h for (px, py) in self.__pins):
            raise ValueError("ERROR: px or py is not valid, {}".format(self.__pins))
        if len(set(self.__pins)) != len(self.__pins):
            raise ValueError("ERROR: repeated pins")

    @property
    def w_orig(self):
        return self.__w

    @property
    def h_orig(self):
        return self.__h

    @property
    def pins_orig(self):
        return self.__pins

    def w(self, angle):
        """
            angle: 0, 1, 2, 3 is 0, 90, 180, 270 degree rotation, clockwise
        """
        if angle == 0 or angle == 2:
            return self.__w
        elif angle == 1 or angle == 3:
            return self.__h
        else:
            raise ValueError("ERROR: angle is invalid to be {}".format(angle))

    def h(self, angle):
        """
            angle: 0, 1, 2, 3 is 0, 90, 180, 270 degree rotation, clockwise
        """
        if angle == 0 or angle == 2:
            return self.__h
        elif angle == 1 or angle == 3:
            return self.__w
        else:
            raise ValueError("ERROR: angle is invalid to be {}".format(angle))

    def pins(self, angle):
        """
            angle: 0, 1, 2, 3 is 0, 90, 180, 270 degree rotation, clockwise
        """
        if angle == 0:
            return self.__pins[:]
        elif angle == 1:
            return [(py, self.__w - 1 - px) for (px, py) in self.__pins]
        elif angle == 2:
            return [(self.__w - 1 - px, self.__h - 1 - py) for (px, py) in self.__pins]
        elif angle == 3:
            return [(self.__h - 1 - py, px) for (px, py) in self.__pins]
        else:
            raise ValueError("ERROR: angle is invalid to be {}".format(angle))

    def show(self, angle):
        """
            show the chiplet with pins
        """
        # plt.imshow(np.ones((self.h(angle), self.w(angle), 3)), cmap="gray")  # white background
        ax = plt.gca()
        ax.set_xlim([0, self.w(angle)])
        ax.set_ylim([0, self.h(angle)])
        plt.gca().set_aspect('equal', adjustable='box')
        
        pins = self.pins(angle)
        cmap = cm.get_cmap("Paired", len(pins))
        colors = cmap(np.linspace(0, 1, len(pins)))
        for idx_p, (px, py) in enumerate(pins):
            ax.add_patch(Rectangle((px, py), 1, 1, color=colors[idx_p]))
            plt.text(px + 0.5, py + 0.5, idx_p, color="w", size=15)
        plt.show()