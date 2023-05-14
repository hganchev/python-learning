import pandas as pd
# uses dataframes for presenting the series of data
# dataframe is using dictionary
x = {'a':[1,2], 'b':[3,4], 'c':[5,6]}

# creating of DataFrame
data = [1,2,3,4,5,6,7]
df = pd.DataFrame(data, index=["one","two","three",'four','five','six','seven'])
print(df)
print(df.iloc[1])
print(df.loc['one'])

# conditions
data = {'ages':[20,30,40,50],'heights':[190,180,170,160]}
df = pd.DataFrame(data)
print(df)
df = df[(df['ages'] > 20) & (df['heights'] < 200)]
print(df)

# returning last rows
print(df.tail(2))

# df.set_index('date', inplace = True) - sets the index of data column
# remove values with df.drop('columnName'), axis=1 - specify only drop a column, 0 - will drop a row,column
# reading files
# df = pd.read_csv('ca-covid.csv')
# df.describe() - returns sumary statistics like std, mean, min, max
# df['culumnName'].value_counts() - return how many values it has, frequency of the values
# df.groupby('columnName1')['columnName2'].sum() - groups the dataset by given column