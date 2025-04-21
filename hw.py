import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics

heartData = pd.read_csv("cleve.mod")

numCols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
catCols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

for col in numCols:
    heartData[col].fillna(heartData[col].median(skipna=True), inplace=True)

for col in catCols:
    heartData[col].fillna(heartData[col].value_counts().idxmax(), inplace=True)

labelEncoder = preprocessing.LabelEncoder()
for col in catCols + ["target"]:
    heartData[col] = labelEncoder.fit_transform(heartData[col])

x = heartData[["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]]
y = heartData["target"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=3)

cls = svm.SVC(kernel="linear")
cls.fit(xTrain, yTrain)
yPred = cls.predict(xTest)

print(metrics.accuracy_score(yTest, yPred))