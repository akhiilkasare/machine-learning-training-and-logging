from imports import *

def preprocessing(data):

    data['Holding_Policy_Duration'].replace('+', '', inplace=True)
    data['Holding_Policy_Duration'].fillna(data['Holding_Policy_Duration'].median(), inplace=True)
    data['Holding_Policy_Type'].fillna(data['Holding_Policy_Type'].median(), inplace=True)

    data = data.drop(['Unnamed: 0', 'ID'],axis=1)


    data = pd.get_dummies(data)

    return data