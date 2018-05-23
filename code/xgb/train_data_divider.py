import numpy as np
from lib import divide_2_batch as d2b

train_path = "/home/leo/ant_leo/data/train_xgb_1.npy"
batch_size = 10
save_path = "/home/leo/ant_leo/data/"

fd = d2b.file_divider(train_path, batch_size)

i = 0

for batch in fd.npy_divider():

    i += 1

    np.save(save_path + "train_batch_{}".format(i), batch)

    print("saved path in %s " %(save_path + "train_batch_{}".format(i)))
