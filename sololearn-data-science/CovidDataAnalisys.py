import pandas as pd
# problem statement: 
# Find the day when the deaths/cases ratio was largest
# To do that first calculte deaths/cases ratio and add it like a column
# to the DataFrame with name 'ratio', then find the largest value
# !hint the output should be a DataFrame, containing all the columns of the dataset
# for the corresponding row
df = pd.read_csv('ca-covid.csv')

df.drop('state',axis=1, inplace=True)
df.set_index('date',inplace=True)
print(df)
print(len(df))
arrRation = []
for i in range(len(df)):
    ratio = df['deaths'][i]/df['cases'][i]
    arrRation.append(ratio)
print(arrRation)
df['ratio'] = arrRation
print(df)
df_max = df[(df['ratio'] == df['ratio'].max())]
print(df_max)
#  second way of doing it
df["ratio"] = df["deaths"] / df["cases"]
max_ratio = df.loc[df["ratio"] == df["ratio"].max()]
print(max_ratio)