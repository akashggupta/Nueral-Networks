import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
c=["NA", "", "#DIV/0!"]
# cleaning the training data set
training= pd.read_csv("C:/Users/user/Documents/pml-training.csv",na_values=c)
training.dropna(axis=1,how='any',inplace=True)
training.drop(training.columns[range(7)], axis=1,inplace =True)

# cleaning the testing data set
testing=pd.read_csv("C:/Users/user/Documents/pml-testing.csv",na_values=c)
testing.dropna(axis=1,how='any',inplace=True)
testing.drop(testing.columns[range(7)], axis=1,inplace =True)

# making the categorical data column "classe" to numerical data
lb_make = LabelEncoder()
training['classe'] = lb_make.fit_transform(training['classe'])

# partitioning the training data into training subset and testing subset
training1, testing1 = train_test_split(training, test_size=0.4)

# preparing the classification model based on random forest
model= RandomForestClassifier()
model.fit(training1[training1.columns[range(52)]], training1['classe'])
prediction=model.predict(testing1[testing1.columns[range(52)]])

# calculating the accuracy
accuracy = metrics.accuracy_score(prediction,testing1['classe'])
print(accuracy)

# Prediction on actual test data set
print(model.predict(testing[testing.columns[range(52)]]))

