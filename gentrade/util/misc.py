import matplotlib.pyplot as plt
import networkx as nx
from deap import gp


def plot_tree(tree, ax):
    nodes, edges, labels = gp.graph(tree)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.drawing.nx_agraph.graphviz_layout(g, prog='dot')
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=400, node_color='lightblue')
    nx.draw_networkx_edges(g, pos, ax=ax)
    nx.draw_networkx_labels(g, pos, labels, ax=ax)
    # plt.show()

