from common import *
from data_load import *

import networkx as nx
import matplotlib.pyplot as plt
import itertools as iter
import random
import pandas as pd
import numpy as np
from tqdm import tqdm,trange
import pprint


# 連想強度データから潜在圏を作る
#def make_assoc_net(source = "source", target = "target", w2v_seed = 0):
#    assoc_data = load_three_metaphor_data(w2v_seed)
#    assoc_net = nx.from_pandas_edgelist(df = assoc_data, source=source, target=target,edge_attr=["weight"], create_using=nx.DiGraph)
#    identity_morphism(assoc_net)
#    return  assoc_net
def make_assoc_net(source = "source", target = "target"):
    assoc_data = load_three_metaphor_data()
    assoc_net = nx.from_pandas_edgelist(df = assoc_data, source=source, target=target,edge_attr=["weight"], create_using=nx.DiGraph)
    identity_morphism(assoc_net)
    return  assoc_net

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
        
#def get_node_data(w2v_seed):
#    node_data = pd.read_csv("./../seed_data/GoogleNews_image_data_seed_{}.tsv".format(w2v_seed),header=None,encoding="utf-8", sep="\t")
#    return  node_data
def get_node_data():
    node_data = pd.read_csv("./../seed_data/similar_butterfly_dancer_3/butterfly_dancer_images.tsv",header=None,encoding="utf-8", sep="\t")
    return  node_data


    
def get_A_B_targets(A,B):
    A_targets = [A]         # 被喩辞
    B_targets = [B]       # 喩辞
    A_fname = [A]  # 被喩辞の英語
    B_fname = [B] # 喩辞の英名
    
    return A_targets, B_targets, A_fname, B_fname
    
#def select_seed_and_f():
def select_f():
    #seed値を選択
#    w2v_seed = input("select seed(0~5)")
    df = pd.read_csv("./../seed_data/noun_data/noun_data.tsv",encoding="utf-8",sep="\t")

    keys = df.columns.values
    f_data = []
    for i in range(10):
        tmp = []
        A = np.random.choice(keys)
        B = np.random.choice(keys)
        tmp.append(A)
        tmp.append(B)
        f_data.append(tmp)
    #選択肢を表示
    pprint.pprint(f_data)

    #使用する射fを選択
    x = int(input("\n select_f (0~9) : "))
    return f_data[x][0], f_data[x][1]
    
def make_node_data(center, node_data):
    center_node_data = list(node_data[node_data[0] == center][1])
    center_node_data.remove(center)
    return center_node_data

#データフレームををコサイン類似度の高い対象間に限定して取り出す関数
#def sort_cossim_data(w2v_seed, target):
def sort_cossim_data(target):
#    data = load_three_metaphor_data(w2v_seed)
    data = load_three_metaphor_data()
    tmp = data[data["source"]== target]
    tmp = tmp.sort_values("weight", ascending=False)
    tmp = tmp[:9]
    tmp = tmp.drop("weight", axis=1)
    tmp = tmp.set_axis([0,1], axis=1)
    return tmp
    
#def sort_cossim_cod_data(w2v_seed, target):
def sort_cossim_cod_data(target):
#    data = sort_cossim_data(w2v_seed, target)
    data = sort_cossim_data(target)
    data = data[1].tolist()
    data.remove(target)
    return data

def remove_identity_image(df):
    index = df.index[df[0] == df[1]]
    return df.drop(index)
    