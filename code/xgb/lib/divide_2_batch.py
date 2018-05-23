import numpy as np


class file_divider():

    def __init__(self, file_path = None, batch_size =None):

        self.file_path = file_path
        self.batch_size = batch_size

    #divide file into batch_size and save in path

    def npy_divider (self):

        #if file_path = None or not endswith xxx :
            #print()
        target_file = np.load(self.file_path)

        constant = int(len(target_file)/self.batch_size)

        for iter_batch in range(0, self.batch_size):

            batch = target_file[iter_batch*constant : (iter_batch+1)*constant]

            # the last batch output all data

            if iter_batch == self.batch_size-1:

                batch = target_file[iter_batch*constant : ]

            print("Batch %s done ..." %(iter_batch+1))

            yield batch
