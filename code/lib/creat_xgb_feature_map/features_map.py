"""
@Authors Leo.cui
7/5/2018
ceate_feature_map

"""

import numpy as np
import pandas as pd


def ceate_feature_map(features, save_path):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()

def get_features(data_path):


    df = pd.read_csv(data_path)

    #df = df.drop(['date','id'], axis=1)

    features = np.arange(1,298)

    #test = pd.DataFrame(data = df, index = None, columns =None)
    #features = df.columns.values
    return features

def main():

    data_path = "/home/leo/ant/model/data/test_a.csv"
    save_path = "/home/leo/ant/model/xgb.fmap"

    features = get_features(data_path)
    ceate_feature_map(features, save_path)

if __name__ == '__main__':
    main()
