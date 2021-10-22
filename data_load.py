import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from multiprocessing import Pool

"""
# 3つの比喩の連想確率のデータをpandas DataFrame 形式で読み込んで返す
#three_metaphor_assoc_data.csvはutf-8で書かれてる
def load_three_metaphor_data():
    DIR = "./../three_metaphor_data/"
    df = pd.read_csv(DIR+"three_metaphor_assoc_data.csv",header=0,index_col=0)
    return df
"""
#w2vデータを連想確率として利用
def load_three_metaphor_data(w2v_seed):
    DIR = "./../seed_data/"
    df = pd.read_csv(DIR+"GoogleNews_cossim_data_seed_{}.tsv".format(w2v_seed), sep ="\t")
    return df
