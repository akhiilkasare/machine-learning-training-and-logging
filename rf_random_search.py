import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import model_selection
from sklearn import metrics

class Hyperparameter:

    df = pd.read_csv("train_folds.csv")

    X = df.drop(['Unnamed: 0', 'ID', 'Response', 'kfold'], axis=1).values
    
    y = df['Response'].values

    classifier = ensemble.RandomForestClassifier(n_jobs=-1)

    ## Defining the parameter grid
    param_grid = {
        "n_estimators": np.arange(100, 1500, 100),
        "max_depth": np.arange(1, 31),
        "criterion": ["gini", "entropy"]
    }


    model = model_selection.RandomizedSearchCV(
        estimator = classifier,
        param_distributions=param_grid,
        n_iter=20,
        scoring='accuracy',
        verbose=10,
        cv=5
    )

    ## Fitting the model to extract best score
    model.fit(X, y)
    print(f"Best score : {model.best_score_}")

    print("Best parameter set :")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")