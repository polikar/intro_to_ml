import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width", "Class"]

#data_frame
df = pd.read_csv(url, header=None, names=column_names)

#dave the data frame to CSV file

df.to_csv("iris_saved.csv", index=False)
df = pd.read_csv("iris_saved.csv")

print("Shape of the Dataframe:", df.shape)
print ("\nFirst 5 rows of the DataFrame")
print(df.head())

print ("\nData types of each column:")
print(df.dtypes)

print("\n Summary statistics of the Dataframe:")
print(df.describe())


