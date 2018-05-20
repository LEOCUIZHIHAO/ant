"""
@Authors Leo.cui
7/5/2018
Format train data

"""

import pandas as pd
import numpy as np

def get_score_format(test_path, answer_sheet_path):
    #read_data
    df = pd.read_csv(test_path)

    answer_sheet = df['id']

    #save answer_sheet csv
    answer_sheet.to_csv(answer_sheet_path, index = None, header = True)

    return


def xbg_format(data_path, save_path, sort_data = True, fillzero = True):

    #read_data
    df = pd.read_csv(data_path)

    #get ride off -1 label
    df = df[(df.label==0)|(df.label==1)]

    #sorting by data
    if sort_data == True:
        df.sort_values('date', inplace = True)

    #delete data column
    df = df.drop(['date','id'], axis=1)

    if fillzero == True:
        #fill na
        df = df.fillna(-999)

    #save csv
    df.to_csv(save_path, index = None, header = False)

    return


def csv2npy(csv_path, npy_path):

    _csv = np.loadtxt(csv_path, delimiter=',')

    #_csv = np.genfromtxt(csv_path, delimiter=",", filling_values = -999)

    np.save(npy_path, _csv)


def main():

    data_path = "/home/leo/ant/model/data/train.csv"
    save_path = "/home/leo/ant/model/data/train_dw.csv"

    #csv_path = "/home/leo/ant/model/data/test_a_xgb_1.csv"
    #npy_path = "/home/leo/ant/model/data/test_a_xgb_1.npy"

    #test_path = "/home/leo/ant/model/data/test_a.csv"
    #answer_sheet_path = "/home/leo/ant/score/answer_sheet.csv"

    xbg_format(data_path, save_path, sort_data = True, fillzero = False)
    #csv2npy(csv_path, npy_path)
    #get_score_format(test_path, answer_sheet_path)

if __name__ == '__main__':
    main()
