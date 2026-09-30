from sklearn.datasets import load_iris                  # the flower dataset
from sklearn.model_selection import train_test_split    # tool that splits data
from sklearn.neighbors import KNeighborsClassifier      # the KNN model

X, y = load_iris(return_X_y=True)                        # X = measurements, y = species number

X_train, X_test, y_train, y_test = train_test_split(     # split into learn + exam
    X, y, test_size=0.2, random_state=1)                 # 20% is the hidden exam

model = KNeighborsClassifier(n_neighbors=5)              # KNN: ask the 5 nearest flowers
model.fit(X_train, y_train)                              # learn from the training flowers

print("Accuracy:", model.score(X_test, y_test))
species = ["setosa", "versicolor", "virginica"]          # the 3 species names
new_flower = [[5.1, 3.5, 1.4, 0.2]]                      # a new flower's 4 measurements
answer = model.predict(new_flower)[0]                    # ask the model (returns 0, 1 or 2)
print("This flower is:", species[answer])    