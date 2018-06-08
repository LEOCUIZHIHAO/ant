"""
@Authors Leo.cui
7/5/2018
Format train data

"""

import pandas as pd
import numpy as np
import os
import time

class Ant_Process_Data(object):
    """docstring for Ant_Process_Data."""

    def xbg_format(data_path, save_path, sort_data = True, fillzero = True):

        last_time  = time.time()

        print("Strat formating train data ...... \n")
        #read_data
        df = pd.read_csv(data_path)
        #sorting by data
        if sort_data == True:
            df.sort_values('date', inplace = True)
        #delete data column
        df = df.drop(['date','id'], axis=1)

        if fillzero == True:
            #fill na
            df = df.fillna(0)
        #save csv
        df.to_csv(save_path, index = None, header = False)

        run_time = time.time() - last_time

        print("Formating time is {} min..\n".format(round(run_time/60),2))

        return

    @staticmethod
    def csv2npy(csv_path, npy_path):

        last_time  = time.time()

        print("Strat converting {} to {} \n".format(csv_path, npy_path))

        _csv = np.loadtxt(csv_path, delimiter=',')

        #_csv = np.genfromtxt(csv_path, delimiter=",", filling_values = -999)
        np.save(npy_path, _csv)

        run_time = time.time() - last_time

        print(".csv 2 .npy time is {} min..\n".format(round(run_time/60),2) )

    @staticmethod
    def merge_file(srcpath, despath):
        '将src路径下的所有文件块合并，并存储到des路径下。'
        files = os.listdir(srcpath)
        #files = ['PU_unlabel.csv', 'train.csv']
        print("\nFind <{}> under the merge path\n".format(files))
        with open(despath, 'w+') as output:
            for eachfile in files:
                print("Merging the <{}> ....\n".format(eachfile))
                filepath = os.path.join(srcpath, eachfile)
                with open(filepath, 'r+') as infile:
                    data = infile.read()
                    output.write(data)
            print("Merge done! Unlabel data added in the original train data\n" + \
                    "\nSaved in path <{}>\n".format(despath))

    @staticmethod
    def split_batch(batch_size, file_path, save_path):

        target_file = np.load(file_path)

        constant = int(len(target_file)/batch_size)

        for iter_batch in range(0, batch_size):

            batch = target_file[iter_batch*constant : (iter_batch+1)*constant]

            # the last batch output all data
            if iter_batch == batch_size-1:

                batch = target_file[iter_batch*constant : ]

            np.save(save_path + "PU_train_batch_{}".format(iter_batch), batch)

            print("saved path in %s " %(save_path + "PU_train_batch_{} with shape <{}>".format(iter_batch, batch.shape)))
