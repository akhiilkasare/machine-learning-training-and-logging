from imports import *
import config

def run(fold):

    ## Reading the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    ## Training data is where kfold is not equal to provided fold
    ## Also we need to reset the index
    df_train = df[df.kfold != fold].reset_index(drop=True)

    ## validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    ## Splitting the data for training
    x_train = df_train.drop(['Response', 'Unnamed: 0', 'ID'], axis=1).values
    y_train = df_train.Response.values

    ## Splitting the data for validation
    x_valid = df_valid.drop(['Response', 'Unnamed: 0', 'ID'], axis=1).values
    y_valid = df_valid.Response.values

    ## Initiating the Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=600, max_depth=6, criterion='gini')

    ## Fitting the model on training data
    clf.fit(x_train, y_train)

    ## Creating predictions for validation samples
    preds = clf.predict(x_valid)

    ## Calculating and printing the evaluation metrics

    accuracy = accuracy_score(y_valid, preds)
    conf_matrix = confusion_matrix(y_valid, preds)
    

    print(f"Fold={fold}, Accuracy={accuracy}, \nConfusion Matrix={conf_matrix}")

    ## Saving the  model
    joblib.dump(
        clf, os.path.join(config.MODEL_OUTPUT, f"df_{fold}.bin")
    )

if __name__ == "__main__":
    run(fold=0)
    run(fold=1)
    run(fold=2)
    run(fold=3)
    run(fold=4)
