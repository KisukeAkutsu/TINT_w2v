from object_established_coslice_simulator import *
from tri_established_coslice_simulator import * 
from analysis import *

w2v_seed, A, B = select_seed_and_f()
established_three_metaphor_sim(w2v_seed, A, B)
all_tri_structure_established_three_metaphor_sim(w2v_seed, A, B)

# 連想確率、TINTのシミュレーション結果(対象同士、三角構造同士)、人間の比喩解釈データをヒートマップで出力する
adj_matrix(w2v_seed,A,B)
object_TINT_edge_correspondence_heatmap(w2v_seed,B,A)


node_data = get_node_data(w2v_seed)
#B_node_data = sort_cossim_data(w2v_seed, source)
B_node_data = sort_cossim_data(w2v_seed, B)
B_node_data = remove_identity_image(B_node_data)
B_init_nodes = list(B_node_data[1])
B_remain_image = [[dom,cod] for dom,cod in iter.permutations(B_init_nodes, 2)]

for tri_dom, tri_cod in B_remain_image:
    tri_TINT_edge_correspondence_heatmap(w2v_seed, B, A, tri_dom, tri_cod)