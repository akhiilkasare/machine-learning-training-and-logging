import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_selection import f_classif


class FeatureSelection:
    
    df = pd.read_csv("train_folds.csv")

    X = df.drop(['Response', 'Unnamed: 0', 'ID'], axis=1)
    col_names = df.columns
    y = df['Response']

    model = ExtraTreesClassifier()
    model.fit(X, y)

    print(model.feature_importances_)

    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)


    feat_importances.nlargest(20).plot(kind='barh')
    feat_importances
    plt.show()



