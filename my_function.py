from common import *
from data_load import *

import networkx as nx
import matplotlib.pyplot as plt
import itertools as iter
import random
import pandas as pd
import numpy as np
from tqdm import tqdm,trange

# 連想強度データから潜在圏を作る
def make_assoc_net(source = "source", target = "target"):
    assoc_data = load_three_metaphor_data()
    assoc_net = nx.from_pandas_edgelist(df = assoc_data, source=source, target=target,edge_attr=["weight"], create_using=nx.DiGraph)
    identity_morphism(assoc_net)
    return  assoc_net


def make_node_data(center, node_data):
    center_node_data = list(node_data[node_data[0] == center][1])
    center_node_data.remove(center)
    return center_node_data

def make_count_matrix(A_node_data, B_node_data, df_edge_corr):
    edge_corr_dict = {(B_node,A_node):0 for A_node in A_node_data for B_node in B_node_data}
    for B_node in B_node_data:
        corr_A_nodes = df_edge_corr[df_edge_corr["B_cod"]==B_node]
        for corr_A in corr_A_nodes.itertuples():
            count = corr_A.count
            edge_corr_dict[(B_node,corr_A.A_cod)] = count
    matrix = list()
    for B_node in B_node_data:
        row = list()
        for A_node in A_node_data+["NA"]:
            row.append(edge_corr_dict[(B_node,A_node)])
        matrix.append(row)