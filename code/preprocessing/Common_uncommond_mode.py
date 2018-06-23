import pandas as pd
import math


def find_mode_list(feature_array):
    modes = pd.DataFrame(columns = ["number", "frequency"])
    for feature in feature_array:
        if (feature in modes["number"].values):
            modes.loc[modes["number"] == feature,"frequency"] += 1
        elif pd.notna(feature):
            modes = modes.append({"number": feature, "frequency": 1}, ignore_index = True)
    modes = modes.sort_values(by=["frequency"], ascending = False).reset_index(drop=True)
    return modes


def main():
    data = pd.read_csv("../../data/black_label_w_missing.csv")
    freq = find_mode_list(data["f20"])
    print(freq)
    print(freq.loc[0].number)

if __name__ == "__main__":
    main()
