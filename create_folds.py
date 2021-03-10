import pandas as pd
from sklearn import model_selection


if __name__ == "__main__":

    # Loading the data
    df = pd.read_csv('cleaned_data.csv')

    # Creating a new column called kfold and filling it with -1
    df['kfold'] = -1

    # Randomizing the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Fetching the data
    X = df.drop('Response', axis=1)

    y = df['Response']

    # Initializing the k-fold class for the model-selection module
    kf = model_selection.StratifiedKFold(n_splits=5)

    # Filling the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    # Saving the new csv with kfold
    df.to_csv("train_folds.csv", index=False)

