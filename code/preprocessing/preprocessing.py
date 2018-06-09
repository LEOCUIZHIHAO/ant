import numpy as np
import pandas as pd
import math
feature_starting_column = 2
save_link = '../../data/replaced_missing_train_complete.csv'

def round_to_whole(data, tolerance):
    p = 10**tolerance
    return int(data*p + 0.5)/p

def replace_missing_by_gaussian(data_array):
    # each column
    print("initiate printing")
    for i in range(data_array.iloc[0,:].size):
        # each feature
        if i < 298:
            continue
        count = 0
        mean = 0
        number_of_values = 0
        # Find mean
        for j in range(data_array.iloc[:,0].size):
            if pd.notna(data_array.iloc[j,i]):
                mean += data_array.iloc[j,i]
                number_of_values += 1
            count += 1
            if count % 5000 == 0:
                print("feature {}".format(i-1) + " processed mean {}".format(count) + " data")
        mean = mean/number_of_values
        print("mean is {}".format(mean))
        # Find the standard deviation
        count = 0
        standard_deviation = 0
        number_of_values = 0
        for j in range(data_array.iloc[:,0].size):
            if pd.notna(data_array.iloc[j,i]):
                standard_deviation += math.pow((data_array.iloc[j,i] - mean),2)
                number_of_values += 1
            count += 1
            if count % 5000 == 0:
                print("feature {}".format(i-1) + " processed standard deviation {}".format(count) + " data")
        standard_deviation = math.sqrt(standard_deviation/number_of_values)
        print("mean: {}".format(mean)+" standard deviation: {}".format(standard_deviation))

        count = 0
        NaN_number = 0
        for j in range(data_array.iloc[:,0].size):
            if pd.isna(data_array.iloc[j,i]):
                # rounding process and random normal are slowing the process down
                number = round_to_whole(np.random.normal(mean,standard_deviation),0)
                if number < 0:
                    data_array.iloc[j,i] = 0
                else:
                    data_array.iloc[j,i] = number
                    NaN_number += 1
            count += 1
            if count % 5000 == 0:
                print("feature {}".format(i-1) + " processed NaN {}".format(count) + " data")
    return data_array

def main():
    data = pd.read_csv('../../data/replaced_missing_train.csv')
    print("data was read")
    data = replace_missing_by_gaussian(data)
    data = data.iloc[:,1:]
    print(data)
    data.to_csv(save_link, index=None)

if __name__ == "__main__":
    main()
