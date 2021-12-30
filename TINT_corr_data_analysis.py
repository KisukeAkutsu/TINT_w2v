import networkx as nx
import pandas as pd
import pprint
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr,spearmanr
from my_function import *

plt.rcParams["font.family"] = "IPAexGothic"

def load_corr_data(fname,target,source):
    Corr_DIR = "./../GraduationThesis/edge_correspondence/"

    #全てのイメージのデータを取得する
    node_data = get_node_data()
    df_edge_corr = pd.read_csv(Corr_DIR+fname,header=0,encoding="utf-8", sep="\t")

    df_edge_corr = df_edge_corr.fillna("NA")
    A_node_data = list(node_data[node_data[0]==target][1])
    B_node_data = list(node_data[node_data[0]==source][1])
    A_node_data.remove(target)
    B_node_data.remove(source)
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

    df = pd.DataFrame(matrix,index=B_node_data,columns=A_node_data+["NA"])
    return df


def make_human_corr_heatmap(data,user_dict,front_dict,main_dict,back_dict):
    # 正しく回答できているデータを分割する(User,前半IMC,メインの対応,後半IMC)
    user_data      = data[list(user_dict.values())]
    front_IMC_data = data[list(front_dict.values())]
    main_corr_data = data[list(main_dict.values())]
    back_IMC_data  = data[list(back_dict.values())]
    print(len(user_data.columns),len(front_IMC_data.columns),len(main_corr_data.columns),len(back_IMC_data.columns))

    val_dict = {"まったく同意しない":1,"あまり同意しない":2,"どちらとも言えない":3,"多少同意する":4,"強く同意する":5}
    # 全く同意しないなどの回答を１〜５へ変換
    rep_value_corr_data = main_corr_data.replace(val_dict)

    #変換したデータの要約をする。ここで平均や標準偏差なんかも計算される
    desc_main_corr_data = rep_value_corr_data.describe()
    print(desc_main_corr_data)
    corr_mtx = []
    std_mtx = []

    T_images = ["舞う" , "飛ぶ" , "花" , "女性" , "空" , "美しさ" , "儚さ" , "羽" , "NA"]
    S_images = ["踊り" , "女性" , "スカート" , "夜" , "音楽" , "回る" , "揺れる" , "舞台"]
    for S_image in S_images:
        row_mean = []
        row_std = []
        for T_image in T_images:
            row_mean.append(desc_main_corr_data[(S_image,T_image)]["mean"])
            row_std.append(desc_main_corr_data[(S_image,T_image)]["std"])
        corr_mtx.append(row_mean)
        std_mtx.append(row_std)

    df = pd.DataFrame(corr_mtx,index = S_images,columns=T_images)
    sns.heatmap(df,vmin=1.0,vmax=5.0,
        cmap="Blues",linewidths=1,cbar=True,
        xticklabels=True,yticklabels=True,
        center=3.0,annot=True)
    plt.ylim(0,8)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    # plt.xlabel("被喩辞の意味を構成する対象")
    # plt.ylabel("喩辞の意味を構成する対象")
    # plt.savefig("butterfly_dancer_human_correspondence.pdf",bbox_inches="tight")

    plt.figure()
    df = pd.DataFrame(std_mtx,index = S_images,columns=T_images)
    sns.heatmap(df,cmap="Blues",linewidths=1,cbar=True,
        xticklabels=True,yticklabels=True,annot=True)
    plt.ylim(0,8)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    # plt.savefig("butterfly_dancer_human_std.pdf")


def show_histgram(data,val_dict):
    rep_value_corr_data = data.replace(val_dict)

    T_images = ["舞う" , "飛ぶ" , "花" , "女性" , "空" , "美しさ" , "儚さ" , "羽" , "NA"]
    S_images = ["踊り" , "女性" , "スカート" , "夜" , "音楽" , "回る" , "揺れる" , "舞台"]

    for S_image in S_images:
        fig = plt.figure(figsize=(10,10))
        for col,T_image in enumerate(T_images):
            counts = rep_value_corr_data[(S_image,T_image)].value_counts()
            data_hist = [counts[i] if i in list(counts.index) else 0 for i in range(1,6)]

            width = 0.8
            x = [i for i in range(1,6)]
            ax = fig.add_subplot(3, 3, col+1,title="{} -> {}".format(S_image,T_image) ,
                xlabel="評価(1:全く同意できない〜5:強く同意する)", xticks=[1,2,3,4,5], ylabel="人数",ylim=(0,30), yticks=[0,10,20,30])

            ax.bar(x, data_hist, width=0.4,tick_label=x)

        plt.tight_layout()
        plt.show()

def make_human_TINT_dist(human_data,TINT_data, val_dict):
    pass

def human_TINT_correlation(human_data,TINT_data,r_function = pearsonr):
    human_one_dist = []
    for idx,row in human_data.iterrows():
        for val in row:
            human_one_dist.append(val)

    TINT_one_dist = []
    for idx,row in TINT_data.iterrows():
        for val in row:
            TINT_one_dist.append(val)
    print("人間とTINTの相関係数 :",r_function(human_one_dist,TINT_one_dist))


def make_human_TINT_each_images_correlation_df(human_data,TINT_data,corr_fname="human_target_data_correlation_all_images.csv",p_value_fname="human_target_data_p_value_all_images.csv",r_function=pearsonr):
    save_dir = "./category/TINT_structure/softmax/"
    corref_matrix = []
    p_value_matrix = []
    for idx,TINT_row in TINT_data.iterrows():
        corref_list = []
        p_value_list = []
        for _,human_row in human_data.iterrows():
            # corref = np.corrcoef(list(human_row),list(TINT_row))[0][1]
            corref, p_value = r_function(list(human_row),list(TINT_row))#ピアソンの相関係数とp値
            corref_list.append(corref)
            p_value_list.append(p_value)
        corref_matrix.append(corref_list)
        p_value_matrix.append(p_value_list)

    S_images = list(TINT_data.index)
    # print(S_images)
    df = pd.DataFrame(corref_matrix,columns=S_images,index=S_images)
    df.to_csv(save_dir+corr_fname)
    df = pd.DataFrame(p_value_matrix,columns=S_images,index=S_images)
    df.to_csv(save_dir+p_value_fname)


def human_data_correlation_non_corr(human_data):
    human_is_corr_list = []
    human_no_corr_list = []
    for idx,row in human_data.iterrows():
        is_corr_sum = 0
        for i,val in enumerate(row):
            if i == 8:
                human_no_corr_list.append(val)
            else:
                is_corr_sum += val
        human_is_corr_list.append(is_corr_sum)
    print("対応づけた回答の総和と対応づかないといった対応の相関 :",np.corrcoef(human_is_corr_list,human_no_corr_list)[0,1])

if __name__ == "__main__":
    data,user_name_dict,front_IMC_name_dict,main_corr_name_dict,back_IMC_name_dict = data_preprocessing(dir="./../../")
    make_human_corr_heatmap(data,user_name_dict,front_IMC_name_dict,main_corr_name_dict,back_IMC_name_dict)
    # print(data.sort_values("age")["age"])
    # print(data.sort_values("sex")["sex"])
    # exit()

    # TODO:対応表ごとのヒストグラムを出す
    # pandasで列ごとのカウント出すやつが確かあった
    # インターンで使ったやつ
    val_dict = {"まったく同意しない":1,"あまり同意しない":2,"どちらとも言えない":3,"多少同意する":4,"強く同意する":5}
    # show_histgram(data,val_dict)

    # TODO:TINTの対応との差分をとったヒートマップを作る
    TINT_corr_data = load_corr_data("Date_all_seed_4000_蝶_踊り子_full_anti_1_iter_1000_correspondence.csv","蝶","踊り子")

    main_corr_data = data[list(main_corr_name_dict.values())]
    rep_value_corr_data = main_corr_data.replace(val_dict)
    # print(rep_value_corr_data)


    desc_main_corr_data = rep_value_corr_data.describe()
    corr_mtx = []
    std_mtx = []
    T_images = ["舞う" , "飛ぶ" , "花" , "女性" , "空" , "美しさ" , "儚さ" , "羽" , "NA"]
    S_images = ["踊り" , "女性" , "スカート" , "夜" , "音楽" , "回る" , "揺れる" , "舞台"]
    for S_image in S_images:
        row_mean = []
        row_std = []
        for T_image in T_images:
            row_mean.append(desc_main_corr_data[(S_image,T_image)]["mean"])
            row_std.append(desc_main_corr_data[(S_image,T_image)]["std"])

        corr_mtx.append(row_mean)
        std_mtx.append(row_std)

    human_corr_data = pd.DataFrame(corr_mtx,index = S_images,columns=T_images)
    human_TINT_correlation(human_corr_data,TINT_corr_data)
    human_corr_data.to_csv("human_correspondence.csv")
    print(human_corr_data)
    print(TINT_corr_data)

    exit()

    r_function = pearsonr
    corr_fname ="human_target_data_{}_correlation_all_images.csv".format(r_function.__name__)
    p_value_fname = "human_target_data_{}_p_value_all_images.csv".format(r_function.__name__)
    make_human_TINT_each_images_correlation_df(human_corr_data,TINT_corr_data,corr_fname=corr_fname,p_value_fname=p_value_fname,r_function=r_function)
    exit()

    conv_TINT_corr_data = []
    for idx,row in TINT_corr_data.iterrows():
        conv_TINT_corr_data.append([val / 250 + 1 for val in row])

    conv_TINT_corr_df = pd.DataFrame(conv_TINT_corr_data,index = S_images,columns=T_images)

    headers = ["","all"]
    tmps = ["0.2","0.5","1.0","2.0","5.0"]
    for header in headers:
        for tmp in tmps:
            if header == "":
                fname = "Softmax_{}_Date_all_seed_4000_蝶_踊り子_full_anti_1_iter_1000_correspondence.csv".format(tmp)
                corr_fname = "softmax_{}_human_target_data_correlation_all_images.csv".format(tmp)
                p_value_fname = "softmax_{}_human_target_data_p_value_all_images.csv".format(tmp)
            else:
                fname = "Softmax_{}_{}_Date_all_seed_4000_蝶_踊り子_full_anti_1_iter_1000_correspondence.csv".format(header,tmp)
                corr_fname = "softmax_{}_{}_human_target_data_correlation_all_images.csv".format(header,tmp)
                p_value_fname = "softmax_{}_{}_human_target_data_p_value_all_images.csv".format(header,tmp)

            TINT_corr_data = load_corr_data(fname,"蝶","踊り子")
            print("softmax_{}_{}：".format(header,tmp),end="")
            # human_TINT_correlation(human_corr_data,TINT_corr_data)
    #         make_human_TINT_each_images_correlation_df(human_corr_data,TINT_corr_data,corr_fname=corr_fname,p_value_fname=p_value_fname)
    exit()

    human_data_correlation_non_corr(human_corr_data)
    exit()

    human_TINT_correlation(human_corr_data,conv_TINT_corr_df)



    sns.heatmap(conv_TINT_corr_df,vmin=1.0,vmax=5.0,
        cmap="Blues",linewidths=1,cbar=True,
        xticklabels=True,yticklabels=True,
        center=3.0,annot=True)
    plt.ylim(0,8)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.savefig("conversion_butterfly_dancer_TINT_correspondence.png")

    plt.figure()
    TINT_human_dist_df = human_corr_data - conv_TINT_corr_df
    sns.heatmap(TINT_human_dist_df,
        cmap="RdBu",linewidths=1,cbar=True,
        xticklabels=True,yticklabels=True,
        center=0,annot=True)
    plt.ylim(0,8)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.savefig("human_TINT_distance_correspondence.png")
