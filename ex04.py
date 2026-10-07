import pandas as pd
from sklearn.model_selection import train_test_split  # Splits data
from sklearn.preprocessing import StandardScaler  # Scales features
from sklearn.neighbors import KNeighborsClassifier  # The KNN model
from sklearn.svm import SVC  # The SVM model

# 1. Load data
data = pd.read_csv("Social_Network_Ads.csv")
X = data[["Age", "EstimatedSalary"]]  # Inputs
y = data["Purchased"]  # Target (1 = buy, 0 = no buy)

# 2. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
)

# 3. Scale features (crucial for distance-based models like KNN and SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Train and evaluate KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("KNN accuracy:", round(knn.score(X_test, y_test), 2))

# 5. Train and evaluate SVM
svm = SVC()
svm.fit(X_train, y_train)
print("SVM accuracy:", round(svm.score(X_test, y_test), 2))

# 6. Predict for a new visitor (Age: 40, Salary: $90,000)
new_person = scaler.transform([[40, 90000]])
answer = knn.predict(new_person)[0]
print("Will they buy? (1=yes, 0=no):", answer)
# )