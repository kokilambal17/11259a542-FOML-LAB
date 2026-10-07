import pandas as pd                                      # tables
from sklearn.cluster import KMeans                        # the grouping model

data = pd.read_csv("Mall_Customers.csv")                 # load the customer CSV
X = data[["Annual Income (k$)","Spending Score (1-100)"]]  # use income and spending

model = KMeans(n_clusters=5, n_init=10, random_state=1)  # ask for 5 groups
model.fit(X)                                             # find the groups
data["Group"] = model.labels_                            # save each customer's group number

print(data["Group"].value_counts())
new_customer = [[75, 85]]                                # income 75k, spends a lot (85/100)
group = model.predict(new_customer)[0]                   # ask which group they belong to
print("This customer belongs to group:", group)
import matplotlib.pyplot as plt          # the charting library

x = data["Annual Income (k$)"]                 # income column
y = data["Spending Score (1-100)"]             # spending column
plt.scatter(x, y, c=model.labels_, cmap="tab10", s=30)  # colour each dot by its group
cen = model.cluster_centers_                   # the 5 group centres
plt.scatter(cen[:,0], cen[:,1], marker="X", s=200, c="black")  # mark the centres
plt.xlabel("income (k$)"); plt.ylabel("spending score")
plt.title("5 customer groups"); plt.show()