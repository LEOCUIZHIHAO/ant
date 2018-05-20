from data import DataSet
from model import GBDT
from tree import f_list
import numpy as np

if __name__ == '__main__':
    data_file = "/home/leo/ant_old/data/train_non_test.csv"
    feature_path = "/home/leo/ant/model/gbdt/gbdt_features/gbdt_features.npy"
    dateset = DataSet(data_file)
    gbdt = GBDT(max_iter=2, sample_rate=0.8, learn_rate=0.5, max_depth=3, loss_type='binary-classification')
    #fit functions parameters are original train data, and a set of all samples
    gbdt.fit(dateset, dateset.get_instances_idset())
    #f_list for all 
    array = np.array(f_list)
    a = array.reshape((2,7))
    b = a[:,-4:]
    np.save(feature_path, b)
