import math
from typing import Dict, List, Tuple

from . import spec


class Block():
    def __init__(self, name: str, area: float, node: int):
        """
            name: only identifier of block
            area: mm^2
            node: the most mature process of manufacturing this block
        """
        self.__name = name
        self.__area = area
        self.__node = node

    @property
    def name(self):
        return self.__name

    @property
    def area(self):
        return self.__area

    @property
    def node(self):
        return self.__node

    def NRE(self):
        return spec.Module_NRE_Cost_Factor[str(self.__node)] * self.__area

    def __eq__(self, __o: object) -> bool:
        return type(self) == type(__o) and self.__name == __o.name

    def __hash__(self) -> int:
        return hash(self.__name)

    def __str__(self):
        return self.__name

    def __repr__(self) -> str:
        return self.__str__()

    def __lt__(self, __o: object):
        return self.__name < __o.name


class Chiplet:
    def __init__(self, blocks: Dict[Block, int], comm: Tuple[float, float] = None):
        """
            blocks[Block instances, number of blocks]
            comm: (input traffic rate, output traffic rate), GB/s
        """
        self.__blocks = dict(blocks)  # cannot change this after initialization
        self.__node = str(min([b.node for b in blocks]))  # chiplet process is decided by the most advanced block

        self.__area_base = 0  # area without D2D
        for blk, num in self.__blocks.items():
            self.__area_base += blk.area * num
        self.__max_perimeter = 2 * (math.sqrt(self.__area_base / spec.max_aspect_ratio) +
                                    math.sqrt(self.__area_base * spec.max_aspect_ratio))
        self.set_comm(comm=comm)

    def set_comm(self, comm: Tuple[float, float] = None):
        self.__comm = comm
        if self.__comm is None:
            self.__area = self.__area_base
        else:
            if spec.D2D_symmetric:
                self.__area_D2D = max(self.__comm) * 2 / spec.D2D_bw_area_density
            else:
                self.__area_D2D = sum(self.__comm) / spec.D2D_bw_area_density
            self.__area = self.__area_base + self.__area_D2D

    @property
    def blocks(self):
        return self.__blocks

    @property
    def area_base(self):
        return self.__area_base

    @property
    def area(self):
        return self.__area

    def __eq__(self, __o: object) -> bool:
        return type(self) == type(__o) and self.__blocks == __o.blocks

    def __hash__(self) -> int:
        """
            Used in set/dict
        """
        return hash(tuple(sorted([(k.name, v) for (k, v) in self.__blocks.items()], key=lambda e: e[0])))

    def __str__(self) -> str:
        return "{}".format(self.__blocks)

    def __repr__(self) -> str:
        return self.__str__()

    def die_yield(self) -> float:
        return (1 + spec.Defect_Density_Die[self.__node] / 100 * self.__area / spec.critical_level)**(
            -spec.critical_level)  # Defect_Density is #/cm2 and area is mm^2

    def N_die_total(self):
        """
            Get the number of all dies sliced from a wafer
        """
        Area_chip = self.area + 2 * spec.scribe_lane * math.sqrt(self.area) + spec.scribe_lane**2
        N_total = math.pi * (spec.wafer_diameter / 2 - spec.edge_loss)**2 / Area_chip - math.pi * (
            spec.wafer_diameter - 2 * spec.edge_loss) / math.sqrt(2 * Area_chip)
        return N_total

    def N_KGD(self):
        """
            Get the number of good dies in dies sliced from a wafer
        """
        return self.N_die_total() * self.die_yield()

    def cost_raw_die(self):
        return spec.Cost_Wafer_Die[self.__node] / self.N_die_total()

    def cost_KGD(self):
        return spec.Cost_Wafer_Die[self.__node] / self.N_KGD()

    def cost_defect(self):
        return self.cost_KGD() - self.cost_raw_die()

    def RE(self):
        return (self.cost_raw_die(), self.cost_defect())

    def NRE(self):
        """
            Total NRE cost for chiplet
        """
        return self.__area * spec.Chip_NRE_Cost_Factor[self.__node] + spec.Chip_NRE_Cost_Fixed[self.__node]


class Package:
    def __init__(self, chiplets: Dict[Chiplet, int]):
        self.chiplets = chiplets

    def __str__(self):
        return "\n".join(["%s:%s" % item for item in self.__dict__.items()])

    def num_chiplets(self) -> int:
        """
            get the total number of chiplets in package, useful for obtaining bonding costs
        """
        return sum(self.chiplets.values())

    def area_chiplets(self) -> float:
        """
            get total area of chiplets
        """
        area = 0
        for chiplet, num in self.chiplets.items():
            assert type(num) == int and num > 0
            area += chiplet.area * num
        assert area > 0
        return area

    def RE(self) -> List[Tuple[float, float, float, float, float]]:
        raise NotImplementedError

    def NRE(self) -> float:
        raise NotImplementedError


class OS(Package):
    """
        Monolithic chip is one chiplet + OS package
    """
    def __init__(self, chiplets: Dict[Chiplet, int]):
        super().__init__(chiplets)

    def area_package(self):
        return self.area_chiplets() * spec.os_area_scale_factor  # get area of organic substrate

    def NRE(self):
        if sum(self.chiplets.values()) == 1:
            factor = 1
        # more layer substrates are used for interconnection
        elif self.area_package() > 30 * 30:  # Large package
            factor = 2
        elif self.area_package() > 17 * 17:
            factor = 1.75
        else:
            factor = 1.5
        return self.area_package() * spec.os_NRE_cost_factor * factor + spec.os_NRE_cost_fixed

    def cost_raw_package(self):
        if sum(self.chiplets.values()) == 1:
            factor = 1
        # more layer substrates are used for interconnection
        elif self.area_package() > 30 * 30:  # Large package
            factor = 2
        elif self.area_package() > 17 * 17:
            factor = 1.75
        else:
            factor = 1.5
        return self.area_package() * spec.cost_factor_os * factor

    def RE(self):
        cost_raw_chiplets = 0
        cost_defect_chiplets = 0
        for chiplet, num in self.chiplets.items():
            cost_raw_chiplets += (chiplet.cost_raw_die() + chiplet.area * spec.c4_bump_cost_factor) * num
            cost_defect_chiplets += chiplet.cost_defect() * num
        cost_defect_package = self.cost_raw_package() * (1 / (spec.bonding_yield_os**self.num_chiplets()) - 1)
        cost_wasted_chiplets = (cost_raw_chiplets + cost_defect_chiplets) * (1 /
                                                                             (spec.bonding_yield_os**self.num_chiplets()) - 1)
        return (cost_raw_chiplets, cost_defect_chiplets, self.cost_raw_package(), cost_defect_package, cost_wasted_chiplets)

    def cost_chiplets(self):
        return sum(self.RE()[0:2])

    def cost_package(self):
        return sum(self.RE()[2:5])

    def cost_total_system(self):
        return sum(self.RE())


class Advanced(Package):
    def __init__(self,
                 chiplets: Dict[Chiplet, int],
                 NRE_cost_factor: float,
                 NRE_cost_fixed: float,
                 wafer_cost: float,
                 defect_density: float,
                 critical_level: int,
                 bonding_yield: float,
                 area_scale_factor: float,
                 chip_last=1):
        super().__init__(chiplets)
        self.NRE_cost_factor = NRE_cost_factor
        self.NRE_cost_fixed = NRE_cost_fixed
        self.wafer_cost = wafer_cost
        self.defect_density = defect_density
        self.critical_level = critical_level
        self.bonding_yield = bonding_yield
        self.area_scale_factor = area_scale_factor
        self.chip_last = chip_last

    def area_interposer(self):
        return self.area_chiplets() * self.area_scale_factor

    def area_package(self):
        return self.area_interposer() * spec.os_area_scale_factor

    def NRE(self):
        return self.area_interposer() * self.NRE_cost_factor + self.NRE_cost_fixed + self.area_package() * spec.cost_factor_os

    def package_yield(self):
        """
            yield of interposer
        """
        return (1 + self.defect_density / 100 * self.area_interposer() / self.critical_level)**(-self.critical_level)

    def N_package_total(self):
        area = self.area_interposer() + 2 * spec.scribe_lane * math.sqrt(self.area_interposer()) + spec.scribe_lane**2
        N_total_package = math.pi * (spec.wafer_diameter / 2 - spec.edge_loss)**2 / area - math.pi * (
            spec.wafer_diameter - 2 * spec.edge_loss) / math.sqrt(2 * area)
        return N_total_package

    def cost_interposer(self):
        """
            raw cost of interposer
        """
        return self.wafer_cost / self.N_package_total() + self.area_interposer() * spec.c4_bump_cost_factor

    def cost_substrate(self):
        return self.area_package() * spec.cost_factor_os

    def cost_raw_package(self):
        return self.cost_interposer() + self.cost_substrate()

    def RE(self):
        """
            RE = chiplet + package
        """
        cost_raw_chiplets = 0
        cost_defect_chiplets = 0
        for chiplet, num in self.chiplets.items():
            cost_raw_chiplets += chiplet.cost_raw_die() * num + chiplet.area * spec.u_bump_cost_factor * num
            cost_defect_chiplets += chiplet.cost_defect() * num
        y1 = self.package_yield()
        y2 = self.bonding_yield**self.num_chiplets()
        y3 = spec.bonding_yield_os
        if self.chip_last == 1:
            cost_defect_package = self.cost_interposer() * (1 / (y1 * y2 * y3) - 1) \
                + self.cost_substrate() * (1 / y3 - 1)
            cost_wasted_chiplets = (cost_raw_chiplets + cost_defect_chiplets) * (1 / (y2 * y3) - 1)

        elif self.chip_last == 0:
            raise ValueError("Error: We do not need chip last")

        assert sum(
            [cost_raw_chiplets, cost_defect_chiplets,
             self.cost_raw_package(), cost_defect_package, cost_wasted_chiplets]) > 0

        return (cost_raw_chiplets, cost_defect_chiplets, self.cost_raw_package(), cost_defect_package, cost_wasted_chiplets)

    def cost_chiplets(self):
        return sum(self.RE()[0:2])

    def cost_package(self):
        return sum(self.RE()[2:5])


class FO(Advanced):
    def __init__(self, chiplets, chip_last=1):
        super().__init__(chiplets, spec.fo_NRE_cost_factor, spec.fo_NRE_cost_fixed, spec.cost_wafer_rdl,
                         spec.defect_density_rdl, spec.critical_level_rdl, spec.bonding_yield_rdl, spec.rdl_area_scale_factor,
                         chip_last)


class SI(Advanced):
    def __init__(self, chiplets):
        super().__init__(chiplets, spec.si_NRE_cost_factor, spec.si_NRE_cost_fixed, spec.cost_wafer_si, spec.defect_density_si,
                         spec.critical_level_si, spec.bonding_yield_si, spec.si_area_scale_factor, 1)
