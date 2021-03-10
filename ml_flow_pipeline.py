from imports import *


if __name__ == "__main__":

    mlflow.set_experiment(experiment_name="MLflow demo")

    print("Loading the data...")
    data = pd.read_csv("cleaned_data.csv")

    X = data.drop(['Response', 'Unnamed: 0', 'ID'], axis=1)
    y = data['Response']

    print("Splitting the data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    ### Buildig the model

    classifier = RandomForestClassifier(n_estimators=600, max_depth=6, criterion='gini')
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)[:,1]

    cm = confusion_matrix(y_test, y_pred)

    model_accuracy = accuracy_score(y_test, y_pred)

    print("Training completed...")
    print("Accuracy : ", model_accuracy)
    print("Confusion matrix : ", cm)

    ## Tracking the model accuracy
    mlflow.log_metric("accuracy", model_accuracy)
    mlflow.sklearn.log_model(classifier, "model")
    


