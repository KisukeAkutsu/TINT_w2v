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
#w2v_cos_sim.csvはshift-jisで書かれてる
def load_three_metaphor_data():
    DIR = "./../word2vec_data/"
    df = pd.read_csv(DIR+"w2v_cos_sim.csv",header=0,index_col=0, encoding='SHIFT-JIS')
    return df
