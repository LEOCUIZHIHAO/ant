import xgboost as xgb
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Imputer, StandardScaler, Normalizer
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
from sklearn.decomposition import PCA

import datetime

def progress_log(names, classifiers):
    now = datetime.datetime.now()
    for name, clf in zip(names, classifiers):
        with open("log/" + "{}_".format(now.year)+"{}_".format(now.month)+"{}".format(now.day), "a") as f:
            f.write(str('*'*10) + str(name) + str('*'*10) + '\n')
            f.write(str(clf) + "\n")

# def progress_log():


def main():
    names = ["random_forest"]
    classifiers = [LogisticRegression(class_weight = "balanced")]
    progress_log(names,classifiers)
    # value = pd.DataFrame(clf)
if __name__ == "__main__":
    main()
