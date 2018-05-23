import csv
import numpy as np

data_path = "/home/alien/Downloads/ant/code/data/xgb_features_0.65.csv"

with open(data_path) as f:
    reader = csv.reader(f)
    rows = [row for row in reader]

#----with this line you can sort your feature, otherwize raw edition    
#rows.sort(key = lambda x:int(x[0]))
temp = []
for data in rows:
    for c in range(int(data[1])):
        temp.append(int(data[0]))
f_l = np.array(temp)
#---------save the numpy as .csv file
#data_test = "/home/alien/Downloads/ant/code/data/DW.csv"
#np.savetxt(data_test, f_l, fmt ='%d', delimiter = ',')

#print(f_l)
