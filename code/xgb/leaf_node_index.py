"""
@Authors Leo.cui
22/5/2018
Xgboost functions

"""

def get_leaf_node_index(bst, dtrain):

    _, _total_leaf_index = bst.get_score(importance_type='weight')

    leaf = bst.predict(dtrain, pred_leaf = True)

    return _total_leaf_index, leaf

def xgb_2_nn_data(_total_leaf_index, leaf, label):

    feed_nn_data = [] # data = train + label

    conncatenate_list = []

    for row in range(len(leaf)):

        for i, x in enumerate(_total_leaf_index):

            leaf_index = leaf[row][i] #leaf_node_index_in_each_tree

            list = [1 if n == leaf_index else 0 for n in x]

            conncatenate_list = conncatenate_list + list

        feed_nn_data.append([label[row], conncatenate_list])
        conncatenate_list = [] #flush conncatenate_list

    return feed_nn_data
