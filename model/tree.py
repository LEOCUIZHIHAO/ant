# -*- coding:utf-8 -*-
from math import log
from random import sample

f_list = []

class Tree:
    def __init__(self):
        self.split_feature = None
        self.leftTree = None
        self.rightTree = None
        # 对于real value的条件为<，对于类别值得条件为=
        # 将满足条件的放入左树
        self.real_value_feature = True
        self.conditionValue = None
        self.leafNode = None

    def get_predict_value(self, instance):
        if self.leafNode:  # 到达叶子节点
            return self.leafNode.get_predict_value()
        if not self.split_feature:
            raise ValueError("the tree is null")
            #value < conditionvalue, put it into leftnode
        if self.real_value_feature and instance[self.split_feature] < self.conditionValue:
            return self.leftTree.get_predict_value(instance)
        #this is for clasification which is the label 
        elif not self.real_value_feature and instance[self.split_feature] == self.conditionValue:
            return self.leftTree.get_predict_value(instance)
        #else for above condition, then put it into rightnode
        return self.rightTree.get_predict_value(instance)

    def describe(self, addtion_info=""):
        if not self.leftTree or not self.rightTree:
            return self.leafNode.describe()
        leftInfo = self.leftTree.describe()
        rightInfo = self.rightTree.describe()
        info = addtion_info+"{split_feature:"+str(self.split_feature)+",split_value:"+str(self.conditionValue)+"[left_tree:"+leftInfo+",right_tree:"+rightInfo+"]}"
        return info


class LeafNode:
    def __init__(self, idset):
        self.idset = idset
        self.predictValue = None

    def describe(self):
        return "{LeafNode:"+str(self.predictValue)+"}"

    def get_idset(self):
        return self.idset

    def get_predict_value(self):
        return self.predictValue

    def update_predict_value(self, targets, loss):
        self.predictValue = loss.update_ternimal_regions(targets, self.idset)


def MSE(values):
    """
    均平方误差 mean square error
    """
    if len(values) < 2:
        return 0
    mean = sum(values)/float(len(values))
    error = 0.0
    for v in values:
        error += (mean-v)*(mean-v)
    return error


def FriedmanMSE(left_values, right_values):
    """
    参考Friedman的论文Greedy Function Approximation: A Gradient Boosting Machine中公式35
    """
    # 假定每个样本的权重都为1
    weighted_n_left, weighted_n_right = len(left_values), len(right_values)
    total_meal_left, total_meal_right = sum(left_values)/float(weighted_n_left), sum(right_values)/float(weighted_n_right)
    diff = total_meal_left - total_meal_right
    return (weighted_n_left * weighted_n_right * diff * diff /
            (weighted_n_left + weighted_n_right))


def construct_decision_tree(dataset, remainedSet, targets, depth, leaf_nodes, max_depth, loss, criterion='MSE', split_points=0):
    if depth < max_depth:
        # todo 通过修改这里可以实现选择多少特征训练
        # to get features, i modify the function in data.py
        # atrributes are the features' name
        attributes = dataset.get_attributes()
        #print(attributes)
        mse = -1
        selectedAttribute = None
        conditionValue = None
        selectedLeftIdSet = []
        selectedRightIdSet = []
        #means get the keys of features' value from feature1 to feature297
        for attribute in attributes:
            #to make sure that the title is the feature title
            is_real_type = dataset.is_real_type_field(attribute)
            # the values are not duplicated, like all value for one feature set.
            attrValues = dataset.get_distinct_valueset(attribute)
            #print(split_points)
            if is_real_type and split_points > 0 and len(attrValues) > split_points:
                attrValues = sample(attrValues, split_points)
            #start from feature 1's all value
            for attrValue in attrValues:
                leftIdSet = []
                rightIdSet = []
                #for all sub train data from feature1 to feature297
                for Id in remainedSet:
                    instance = dataset.get_instance(Id)
                    value = instance[attribute]
                    # 将满足条件的放入左子树 <=
                    if (is_real_type and value < attrValue)or(not is_real_type and value == attrValue):
                        #add samples'key value
                        leftIdSet.append(Id)
                    else:
                        rightIdSet.append(Id)
                #targets is the loss function
                #here contains each train sample's redisual value of label, which is the right or wrong
                leftTargets = [targets[id] for id in leftIdSet]
                rightTargets = [targets[id] for id in rightIdSet]
                #calculate the sum_mse for current value of the feature.
                sum_mse = MSE(leftTargets)+MSE(rightTargets)
                #if < cureent mse, means the mse is better, so the selectedAttribute = current attribute which is the feature
                if mse < 0 or sum_mse < mse:
                    selectedAttribute = attribute
                    #the conditionvalue is the feature's current value in unique set.
                    conditionValue = attrValue
                    mse = sum_mse
                    selectedLeftIdSet = leftIdSet
                    selectedRightIdSet = rightIdSet
        if not selectedAttribute or mse < 0:
            raise ValueError("cannot determine the split attribute.")
        tree = Tree()
        tree.split_feature = selectedAttribute
        #print(selectedAttribute)
        f_list.append(selectedAttribute)
        #print(f_list)
        tree.real_value_feature = dataset.is_real_type_field(selectedAttribute)
        tree.conditionValue = conditionValue
        #here is the seperate for the tree
        tree.leftTree = construct_decision_tree(dataset, selectedLeftIdSet, targets, depth+1, leaf_nodes, max_depth, loss)
        tree.rightTree = construct_decision_tree(dataset, selectedRightIdSet, targets, depth+1, leaf_nodes, max_depth, loss)
        #print(tree.f_list)
        return tree
    else:  # 是叶子节点
        node = LeafNode(remainedSet)
        node.update_predict_value(targets, loss)
        leaf_nodes.append(node)
        tree = Tree()
        tree.leafNode = node
        return tree
