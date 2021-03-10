from imports import *
from logs import log
from preprocessing import *

## Setting up a logger file
logger = log(path='/home/akhil/Downloads/deep_learning/analytics-vidhya/logs', file="cross_val.logs")

## Loading the dataset
data = pd.read_csv("/home/akhil/Downloads/deep_learning/analytics-vidhya/cleaned_data.csv")

## Preprocessing the data

# data = preprocessing(data)

## Splitting the data into train and testing

X = data.drop(['Response', 'Unnamed: 0', 'ID'], axis=1)
y = data['Response']


## Creating a dictionary of classifiers

models = {
    "KNN": KNeighborsClassifier(),
    "RF": RandomForestClassifier(),
    "GB": GradientBoostingClassifier(),
    "DTC": DecisionTreeClassifier(),
    "BC": BaggingClassifier(),
    "XGB": XGBClassifier(),
    "EXT": ExtraTreesClassifier(),
    "LG": LogisticRegression(),
    "BBC": BalancedBaggingClassifier(),
    "EEC": EasyEnsembleClassifier(),
}

logger.info("Start Cross Validation")

for model_name, model in models.items():
    logger.info("Train {}".format(model_name))

    ## Cross val score for each classifier
    scores = cross_val_score(model, X, y, scoring='accuracy')

    logger.info("The mean score for {}: {:.3f}".format(model_name, scores.mean()))

    logger.info("-------------------------------")

logger.info("Cross Validation Ends")
