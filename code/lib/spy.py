#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 18:58:24 2018

@author: alien
"""

from random import sample
import csv

sample_rate = 0.15

#set of black label data
data_path_b = "/home/alien/Downloads/ant/code/data/black_label.csv"
#set of P-s data
data_path_nb = "/home/alien/Downloads/ant/code/data/new_black_label.csv"

#s data with rate 0.15
data_path_s = "/home/alien/Downloads/ant/code/data/DW.csv"

#U data
#data_path_u = "/home/alien/Downloads/ant/code/data/new_unlabel.csv"

#add spy data to the unlabel data as a new U+s data.
data_path_nu = "/home/alien/Downloads/ant/code/data/spy_U.csv"


with open(data_path_b) as f:
    reader = csv.reader(f)
    rows = [row for row in reader]


temp1 = list(rows)
# selected data.
temp = sample(temp1, int(len(rows)*sample_rate))
#P-s data.
temp2 = []
for tip in temp1:
    if tip in temp:
        continue
    else:       
        temp2.append(tip)
with open(data_path_s, "w") as f:
    writer = csv.writer(f)
    writer.writerows(temp)

with open(data_path_nb, "w") as f:
    writer = csv.writer(f)
    writer.writerows(temp2)

with open(data_path_nu, "a+") as f:
    writer = csv.writer(f)
    writer.writerows(temp)   
#with open(data_path_nu, "a+") as f:
#    f.write(open(data_path_u).read())
#    f.write(open(data_path_s).read())
 