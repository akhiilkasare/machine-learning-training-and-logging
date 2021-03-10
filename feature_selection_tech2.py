import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


class FeatureSelectionFromModl:
    
    df = pd.read_csv("train_folds.csv")

    X = df.drop(['Response', 'Unnamed: 0', 'ID'], axis=1)
    col_names = df.columns
    y = df['Response']

    ## Initializing the model
    model = RandomForestClassifier()

    ## Selecting from the model
    sfm = SelectFromModel(estimator=model)
    X_transformed = sfm.fit_transform(X, y)
    
    ## Checking which features are selected
    support = sfm.get_support()

    ## Getting feature names
    print([x for x, y in zip(col_names, support) if y == True])
