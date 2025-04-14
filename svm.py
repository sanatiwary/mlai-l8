# support vector machine
# classification algorithms: logistical regression, decision tree, random forest
# svm used in both classification and regression
# scalar - magnitude - ex. 5m/s
# vector - magnitude and direction - ex 5m/s to the north

import pandas as pd
from sklearn import datasets
from sklearn import svm
from sklearn import metrics # for accuracy
from sklearn.model_selection import train_test_split

cancerDict = datasets.load_breast_cancer()
print(cancerDict.keys())

cancerData = pd.DataFrame(cancerDict.data)
cancerData.columns = cancerDict.feature_names # input
cancerData["isCancer"] = cancerDict.target # output

print(cancerData.info())
print(cancerData.head())

y = cancerData["isCancer"]
cancerData.drop("isCancer", axis=1)

X = cancerData

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=3)

cls = svm.SVC(kernel="linear")
cls.fit(xTrain, yTrain)
yPred = cls.predict(xTest)

print("accuracy score: ", metrics.accuracy_score(yTest, yPred))