from sklearn.datasets import load_breast_cancer          # the tumour dataset
from sklearn.model_selection import train_test_split     # splits data
from sklearn.tree import DecisionTreeClassifier          # one tree
from sklearn.ensemble import RandomForestClassifier      # a forest of trees
import matplotlib.pyplot as plt                          # the charting library

# 1. Load and split the tumour dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=1)

# 2. Train the models
tree = DecisionTreeClassifier(random_state=1)
tree.fit(X_train, y_train)

forest = RandomForestClassifier(n_estimators=100, random_state=1)
forest.fit(X_train, y_train)

# 3. Print evaluation scores
print("One tree accuracy:", round(tree.score(X_test, y_test), 3))
print("Forest accuracy :", round(forest.score(X_test, y_test), 3))

# 4. Predict a new instance
names = data.target_names
new_tumour = [X_test[0]]
result = forest.predict(new_tumour)[0]
print("Diagnosis:", names[result])
print("-" * 40)

# 5. Extract and sort features specifically from the 'forest' model
importances = forest.feature_importances_       # Changed 'model' to 'forest'
names = data.feature_names
top = sorted(zip(importances, names), reverse=True)[:5]

# 6. Unpack and plot top 5 features
vals = [t[0] for t in top]
labs = [t[1] for t in top]

plt.figure(figsize=(8, 4.5))                             # Prevents long labels from clipping
plt.barh(labs[::-1], vals[::-1], color="#2F49D1")
plt.xlabel("importance")
plt.title("Top 5 features (Random Forest)")

plt.tight_layout()
plt.show()
