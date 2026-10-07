import pandas as pd                                      # tables
from sklearn.cluster import AgglomerativeClustering       # the tree-based grouping model

data = pd.read_csv("Mall_Customers.csv")                 # load the customer CSV
X = data[["Annual Income (k$)","Spending Score (1-100)"]]  # use income and spending

model = AgglomerativeClustering(n_clusters=5)            # ask for 5 groups from the tree
data["Group"] = model.fit_predict(X)                     # find groups + save each customer's group

print(data.groupby("Group")[["Annual Income (k$)","Spending Score (1-100)"]].mean())  # average of each group
new_customer = [80, 15]                                  # income 80k, spends very little (15/100)
averages = data.groupby("Group")[["Annual Income (k$)","Spending Score (1-100)"]].mean()  # each group's centre
distances = ((averages - new_customer) ** 2).sum(axis=1)  # distance to each group's centre
print("Closest group:", distances.idxmin())
