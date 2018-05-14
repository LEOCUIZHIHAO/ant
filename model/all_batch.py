import csv
i = 1
data_path = "/home/alien/Downloads/ant/code/data/rf_data/train_non_test.csv"
data = "/home/alien/Downloads/ant/code/data/rf_data/"
#currently especially for the first line
line_cnt = 0
#for the first row which are date, label, etc.
title = []

fname =[i for i in range(1,2)]
for it in fname:
    i_str = str(i)
    batch_file = data+ i_str + '.csv'
    with open(data_path) as f:
        for line in f:
            if line_cnt == 0:
                title.append(line)
                line_cnt += 1
            else:
                with open(batch_file,"a") as mon:
                    if i <= 400:
                        mon.write(line)
                        i += 1
                    else:
                        mon.close()
                