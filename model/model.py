# -*- coding:utf-8 -*-
from datetime import datetime
import abc
from random import sample
from math import exp, log
from tree import construct_decision_tree


class RegressionLossFunction(metaclass=abc.ABCMeta):
    def __init__(self, n_classes):
        self.K = n_classes

    @abc.abstractmethod
    def compute_residual(self, dataset, subset, f):
        """计算残差"""

    @abc.abstractmethod
    def update_f_value(self, f, tree, leaf_nodes, subset, dataset, learn_rate, label=None):
        """更新F_{m-1}的值"""

    @abc.abstractmethod
    def initialize(self, f, dataset):
        """初始化F_{0}的值"""

    @abc.abstractmethod
    def update_ternimal_regions(self, targets, idset):
        """更新叶子节点的返回值"""


class LeastSquaresError(RegressionLossFunction):
    """用于回归的最小平方误差损失函数"""
    def __init__(self, n_classes):
        if n_classes != 1:
            raise ValueError("``n_classes`` must be 1 for regression but "
                             "was %r" % n_classes)
        super(LeastSquaresError, self).__init__(n_classes)

    def compute_residual(self, dataset, subset, f):
        residual = {}
        for id in subset:
            #figure labe here 513
            if dataset.get_instance(id)['label'] == 0:
                y_i = -1
            else:
                y_i = 1
            #y_i = dataset.get_instance(id)['label']
            residual[id] = y_i - f[id]
        return residual

    def update_f_value(self, f, tree, leaf_nodes, subset, dataset, learn_rate, label=None):
        data_idset = set(dataset.get_instances_idset())
        subset = set(subset)
        for node in leaf_nodes:
            for id in node.get_idset():
                f[id] += learn_rate*node.get_predict_value()
        for id in data_idset-subset:
            f[id] += learn_rate*tree.get_predict_value(dataset.get_instance(id))

    def initialize(self, f, dataset):
        """初始化F0，我们可以用训练样本的所有值的平均值来初始化，为了方便，这里初始化为0.0"""
        ids = dataset.get_instances_idset()
        for id in ids:
            f[id] = 0.0

    def update_ternimal_regions(self, targets, idset):
        sum1 = sum([targets[id] for id in idset])
        return sum1/len(idset)


class ClassificationLossFunction(metaclass=abc.ABCMeta):
    """分类损失函数的基类"""
    def __init__(self, n_classes):
        self.K = n_classes

    @abc.abstractmethod
    def compute_residual(self, dataset, subset, f):
        """计算残差"""

    @abc.abstractmethod
    def update_f_value(self, f, tree, leaf_nodes, subset, dataset, learn_rate, label=None):
        """更新F_{m-1}的值"""

    @abc.abstractmethod
    def initialize(self, f, dataset):
        """初始化F_{0}的值"""

    @abc.abstractmethod
    def update_ternimal_regions(self, targets, idset):
        """更新叶子节点的返回值"""

# target binomialdeviance we are focusing on
class BinomialDeviance(ClassificationLossFunction):
    """二元分类的损失函数"""
    def __init__(self, n_classes):
        if n_classes != 2:
            raise ValueError("{0:s} requires 2 classes.".format(
                self.__class__.__name__))
        super(BinomialDeviance, self).__init__(1)

    def compute_residual(self, dataset, subset, f):
        #is that right to calculate loss value?
        residual = {}
        for id in subset:
            if dataset.get_instance(id)['label'] == 0:
                y_i = -1
            else:
                y_i = 1
            #y_i = dataset.get_instance(id)['label']
            residual[id] = 2.0*y_i/(1+exp(2*y_i*f[id]))
        return residual

    def update_f_value(self, f, tree, leaf_nodes, subset, dataset, learn_rate, label=None):
        data_idset = set(dataset.get_instances_idset())
        subset = set(subset)
        for node in leaf_nodes:
            for id in node.get_idset():
                f[id] += learn_rate*node.get_predict_value()
        for id in data_idset-subset:
            f[id] += learn_rate*tree.get_predict_value(dataset.get_instance(id))
# all initialization for each sample f[id] = 0
    def initialize(self, f, dataset):
        #keys are 1~2764
        ids = dataset.get_instances_idset()
        for id in ids:
            f[id] = 0.0

    def update_ternimal_regions(self, targets, idset):
        sum1 = sum([targets[id] for id in idset])
        if sum1 == 0:
            return sum1
        sum2 = sum([abs(targets[id])*(2-abs(targets[id])) for id in idset])
        return sum1 / sum2
#for this part we dont care


class GBDT:
    def __init__(self, max_iter, sample_rate, learn_rate, max_depth, loss_type='multi-classification', split_points=0):
        self.max_iter = max_iter
        self.sample_rate = sample_rate
        self.learn_rate = learn_rate
        self.max_depth = max_depth
        self.loss_type = loss_type
        self.split_points = split_points
        self.loss = None
        self.trees = dict()

    def fit(self, dataset, train_data):
        if self.loss_type == 'multi-classification':
            print("useless")

        else:
            if self.loss_type == 'binary-classification':
                #first step: loss = get_label_size 
                self.loss = BinomialDeviance(n_classes=dataset.get_label_size())
            elif self.loss_type == 'regression':
                self.loss = LeastSquaresError(n_classes=1)

            f = dict()  # 记录F_{m-1}的值
            self.loss.initialize(f, dataset)
            #here is the iter times
            for iter in range(1, self.max_iter+1):
                #the train_data is a set of keys which are 1,2,3,4...2764
                subset = train_data
    
                if 0 < self.sample_rate < 1:
                    #from subset select 0.8* 2764 samples randomly
                    subset = sample(subset, int(len(subset)*self.sample_rate))
                # 用损失函数的负梯度作为回归问题提升树的残差近似值
                # this value is for binary clasification tree loss function. need to check.
                # a set for each samples' label the cost function
                residual = self.loss.compute_residual(dataset, subset, f)
                leaf_nodes = []
                #the targets loss value set
                targets = residual
                #construct tree,the depth value is 0? for the first time?
                tree = construct_decision_tree(dataset, subset, targets, 0, leaf_nodes, self.max_depth, self.loss, self.split_points)
                #this is for number of trees
                self.trees[iter] = tree
                #print(tree.split_feature)
                self.loss.update_f_value(f, tree, leaf_nodes, subset, dataset, self.learn_rate)
                if isinstance(self.loss, RegressionLossFunction):
                    # todo 判断回归的效果
                    pass
                else:
                    train_loss = self.compute_loss(dataset, train_data, f)
                    print("iter%d : train loss=%f" % (iter,train_loss))
                    #print(d_list)
    def compute_loss(self, dataset, subset, f):
        loss = 0.0
        if self.loss.K == 1:
            for id in dataset.get_instances_idset():
                if dataset.get_instance(id)['label'] == 0:
                    y_i = -1
                else:
                    y_i = 1
                #y_i = dataset.get_instance(id)['label']
                f_value = f[id]
                p_1 = 1/(1+exp(-2*f_value))
                try:
                    loss -= ((1+y_i)*log(p_1)/2) + ((1-y_i)*log(1-p_1)/2)
                except ValueError as e:
                    print(y_i, p_1)
        else:
            for id in dataset.get_instances_idset():
                instance = dataset.get_instance(id)
                f_values = f[id]
                exp_values = {}
                for label in f_values:
                    exp_values[label] = exp(f_values[label])
                probs = {}
                for label in f_values:
                    probs[label] = exp_values[label]/sum(exp_values.values())
                    # 预测的越准确则log(probs[instance["label"]])越接近0 loss也就越小
                loss -= log(probs[instance["label"]])
                #compare with each sample,determian the loss value for each sample, then sum and devided by data size.
        return loss/dataset.size()

    def compute_instance_f_value(self, instance):
        """计算样本的f值"""
        if self.loss.K == 1:
            f_value = 0.0
            for iter in self.trees:
                f_value += self.learn_rate * iter.get_predict_value(instance)
        else:
            f_value = dict()
            for label in self.loss.labelset:
                f_value[label] = 0.0
            for iter in self.trees:
                # 对于多分类问题，为每个类别构造一颗回归树
                for label in self.loss.labelset:
                    tree = self.trees[iter][label]
                    f_value[label] += self.learn_rate*tree.get_predict_value(instance)
        return f_value

    def predict(self, instance):
        """
        对于回归和二元分类返回f值
        对于多元分类返回每一类的f值
        """
        return self.compute_instance_f_value(instance)

    def predict_prob(self, instance):
        """为了统一二元分类和多元分类，返回属于每个类别的概率"""
        if isinstance(self.loss, RegressionLossFunction):
            raise RuntimeError('regression problem can not predict prob ')
        if self.loss.K == 1:
            f_value = self.compute_instance_f_value(instance)
            probs = dict()
            probs['+1'] = 1/(1+exp(-2*f_value))
            probs['-1'] = 1 - probs['+1']
        else:
            f_value = self.compute_instance_f_value(instance)
            exp_values = dict()
            for label in f_value:
                exp_values[label] = exp(f_value[label])
            exp_sum = sum(exp_values.values())
            probs = dict()
            # 归一化，并得到相应的概率值
            for label in exp_values:
                probs[label] = exp_values[label]/exp_sum
        return probs

    def predict_label(self, instance):
        """预测标签"""
        predict_label = None
        if isinstance(self.loss, BinomialDeviance):
            probs = self.predict_prob(instance)
            #care about this value
            predict_label = 1 if probs[1] >= probs[-1] else -1
        else:
            probs = self.predict_prob(instance)
            # 选出K分类中，概率值最大的label
            for label in probs:
                if not predict_label or probs[label] > probs[predict_label]:
                    predict_label = label
        return predict_label
