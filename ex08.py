import pandas as pd                                      # tables
from sklearn.model_selection import train_test_split     # splits data
from sklearn.ensemble import GradientBoostingClassifier  # the boosting model

data = pd.read_csv("heart(4).csv")                          # load the patient CSV
X = data.drop("target", axis=1)                          # inputs: all columns except the answer
y = data["target"]                                       # answer: 1 disease, 0 no disease

X_train, X_test, y_train, y_test = train_test_split(     # split learn + exam
    X, y, test_size=0.2, random_state=1)

model = GradientBoostingClassifier()                     # make the booster
model.fit(X_train, y_train)                              # train it
print("Accuracy:", round(model.score(X_test, y_test), 3))  # sco
new_patient = [X_test.iloc[0]]                           # one new patient's test results
result = model.predict(new_patient)[0]                   # predict (1 or 0)
print("Heart disease? (1=yes, 0=no):", result)