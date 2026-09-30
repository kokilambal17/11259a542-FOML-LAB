import pandas as pd                                     # pandas = tables in Python
import seaborn as sns                                   # seaborn = gives us the Titanic data

data = sns.load_dataset("titanic")                      # load the Titanic passenger table
data = data[["survived","pclass","sex","age","fare","embarked"]]  # keep only 6 useful columns

print(data.isnull().sum())                              # count the empty cells in each column

data["age"] = data["age"].fillna(data["age"].median())  # fill empty ages with the middle age
data["embarked"] = data["embarked"].fillna("S")         # fill empty ports with "S" (most common)

data = pd.get_dummies(data, columns=["sex","embarked"], dtype=int)  # turn words into 0/1 columns

print(data.head())                                      # show the first 5 rows of the clean table
print("Any blanks left?", data.isnull().values.any())   # check: False means fully clean
new = pd.DataFrame([{"pclass":2,"sex":"female","age":27,"fare":30,"embarked":"C"}])  # one new passenger
new = pd.get_dummies(new, columns=["sex","embarked"], dtype=int)     # clean it the same way
new = new.reindex(columns=data.drop("survived", axis=1).columns, fill_value=0)  # match the columns
print(new)