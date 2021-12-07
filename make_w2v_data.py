#プログラムを組む時に使う実験用ファイル
from utility import *

import gensim.models
import random
import numpy as np
import pandas as pd

model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

#import gensim.downloader as api
#model = api.load("glove-wiki-gigaword-300")

#KeyedValueからDictに変更
vector_dict = {}
for word in model.index2word:
    vector_dict[word] = model.get_vector(word)

#Dictからdfに変換
df = pd.DataFrame.from_dict(vector_dict)


#乱数ごとに2000データずつ作成&csvに保存
for i in range(5):
    df_sample = df.sample(n=100, axis=1, random_state= i)
    df_sample.to_csv("./../seed_data/GoogleNews_w2v_data_seed_{}.tsv".format(i), sep = "\t")


    #乱数ごとに2000データずつ作成したcsvファイルを作成
for i in range(5):
    df_sample = df.sample(n=100, axis=1, random_state= i)
    df_sample.to_csv("./../seed_data/GoogleNews_w2v_data_seed_{}.tsv".format(i), sep="\t")

    #各seedごとのコサイン類似度を計算したcsvファイルを作る
for i in range(5):
    df_seed = df.sample(n=100, axis=1, random_state=i)
    keys = df_seed.columns.values
    lst = []
    for key1 in keys:
        for key2 in keys:
            tmp = []
            cossim = np.dot(model[key1],model[key2])/(np.linalg.norm(model[key1])*np.linalg.norm(model[key2]))
            if cossim < 0:
                cossim = 0
            tmp.append(key1)
            tmp.append(key2)
            tmp.append(cossim)
            lst.append(tmp)
    cossim_df = pd.DataFrame(lst)
    cossim_df = cossim_df.set_axis(["source","target","weight"],axis="columns")
    df_seed.to_csv("./../seed_data/GoogleNews_w2v_data_seed_{}.tsv".format(i), index=False, sep="\t")
    cossim_df = cossim_df.to_csv("./../seed_data/GoogleNews_cossim_data_seed_{}.tsv".format(i), index=False, sep="\t")

    
for i in range(5):
    df_seed = df.sample(n=100, axis=1, random_state=i)
    keys = df_seed.columns.values
    lst = []
    for key1 in keys:
        for key2 in keys:
            tmp = []
            tmp.append(key1)
            tmp.append(key2)
            lst.append(tmp)
    image_df = pd.DataFrame(lst)
    image_df.to_csv("./../seed_data/GoogleNews_image_data_seed_{}.tsv".format(i), index=False, sep="\t")

    

