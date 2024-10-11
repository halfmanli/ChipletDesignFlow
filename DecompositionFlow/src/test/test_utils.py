import networkx as nx
from ..model import Block
from .. import utils
import os

dir_chaco = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../tool/chaco")

if __name__ == "__main__":
    # g = nx.DiGraph()
    # g.add_edge(0, 1, comm=4)
    # utils.mincut_chaco(dir_chaco=dir_chaco, G=g, k=2, clean=False)
    utils.check_placer()
    # utils.place(areas=[4, 4, 4], connection_matrix=[[0, 1, 1], [0, 0, 0], [0, 0, 0]])
