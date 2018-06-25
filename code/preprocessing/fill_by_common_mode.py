import pandas as pd
import math


# def find_mode_list(feature_array):
#     print("Initiate finding mode")
#     # Create data frame object
#     modes = pd.DataFrame(columns = ["number", "frequency"])
#     # For each of the feature
#     number_of_nan = 0
#     for feature in feature_array:
#         # If value is found inside the modes list, then add one to its frequency
#         if (feature in modes["number"].values):
#             modes.loc[modes["number"] == feature,"frequency"] += 1
#         elif pd.notna(feature):
#             modes = modes.append({"number": feature, "frequency": 1}, ignore_index = True)
#         else:
#             number_of_nan += 1
#     modes = modes.sort_values(by=["frequency"], ascending = False).reset_index(drop=True)
#     modes["frequency"] = modes["frequency"]/(len(feature_array) - number_of_nan)
#     print("End of finding mode")
#     return modes

def find_common_mode(black_frequency_list, white_frequency_list):
    # The function takes in two data frame series as input and outputs the common mode between two series
    # print("Initiate find common_mode")
    # iterate through top 5 of each of the mode and find the mode with least frequency difference
    min_freq_diff = 99999;
    min_mode_value = 0;
    for i in range(5):
        print(black_frequency_list)
        print("index i:{}".format(i))
        if i == len(white_frequency_list)-1:
            break;
        for j in range(5):
            print("index j:{}".format(j))
            if j == len(black_frequency_list)-1:
                break;
            # Calculate the difference in frequency, the number with smallest frequency is stored in min_mode_value
            freq_diff = abs((white_frequency_list.iloc[i] - black_frequency_list.iloc[j])*100)
            if (freq_diff < min_freq_diff) and (white_frequency_list.index[i] == black_frequency_list.index[j]):
                min_mode_value = white_frequency_list.index[i]
    common_mode = min_mode_value
    # print(black_frequency_list)
    # print("*******************")
    # print(white_frequency_list)
    # print("*******************")
    # print("End of finding common mode, mode is {}".format(common_mode))
    return common_mode

def replace_missing_by_custom_mode(black_data,white_data,test_data):
    print("Initiate custom mode filling nan process")
    for i in range(1, 297):
        if black_data["f{}".format(i)].mode()[0] == white_data["f{}".format(i)].mode()[0]:
            common_mode = black_data["f{}".format(i)].mode()[0]
        else:
            # Calculate a list of occurrence in each feature
            black_mode_list = black_data["f{}".format(i)].value_counts() / len(black_data["f{}".format(i)])
            white_mode_list = white_data["f{}".format(i)].value_counts() / len(white_data["f{}".format(i)])
            # Calculate the common occurrence between black and white data
            common_mode = find_common_mode(black_mode_list,white_mode_list)
        black_data["f{}".format(i)] = black_data["f{}".format(i)].fillna(black_data["f{}".format(i)].mode()[0])
        white_data["f{}".format(i)] = white_data["f{}".format(i)].fillna(white_data["f{}".format(i)].mode()[0])
        test_data["f{}".format(i)] = test_data["f{}".format(i)].fillna(common_mode)
        print("Filled feature {}".format(i) + "***Black Filled:{}".format(black_data["f{}".format(i)].mode()[0]) + "***White Filled:{}".format(white_data["f{}".format(i)].mode()[0]) + "***Test Filled:{}".format(common_mode))
        print("******************************")
    print("End of custom mode filling")
    return black_data, white_data, test_data

def main():
    black_data = pd.read_csv("../../data/black_label_w_missing.csv")
    print("loaded black data")
    white_data = pd.read_csv("../../data/white_label_w_missing.csv")
    print("loaded white data")
    test_data = pd.read_csv("../../data/test_a.csv")
    print("loaded test data")
    black_data_filled, white_data_filled, test_data_filled = replace_missing_by_custom_mode(black_data,white_data,test_data)
    black_data_filled.to_csv("black_data_w_mode_filled.csv")
    print("saved black data")
    white_data_filled.to_csv("white_data_w_mode_filled.csv")
    print("saved white data")
    test_data_filled.to_csv("test_data_w_mode_filled.csv")
    print("saved test data")
    print("end of program")

if __name__ == "__main__":
    main()
