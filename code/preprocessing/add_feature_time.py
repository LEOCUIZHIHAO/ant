import pandas as pd
import numpy as np

train_path = ""
new_train_save_path = ""
test_path = ""
new_test_save_path = ""

def convert_to_date(array):
    list = []
    for i in range(len(array)):
        value = array[i] % 100
        list.append(value)
    return list

def main():
    train_data = pd.read_csv(train_path)
    train_data_new_feature = convert_to_date(train_data["date"])
    train_data['f298'] = train_data_new_feature
    train_data.to_csv(new_train_save_path)

    test_data = pd.read_csv(test_path)
    test_data_new_feature = convert_to_date(test_data["date"])
    test_data['f298'] = test_data_new_feature
    test_data.to_csv(new_test_save_path)
main()
