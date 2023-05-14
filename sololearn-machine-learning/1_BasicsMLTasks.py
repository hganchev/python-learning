# Supervised - we have a known target based on past data
# - Classification and Regression problems
# - Logistic Regression, Decision Trees, Random Forests, Neural Networks
# Unsupervised - there isn't a past data

# the mean and median is called averages
# the median can also be throght the 50th percentile 
# this means that 50% are less than the median and 50 are greater
# it also has 25th(25< and 75>) and 75th(25> and 75<)
import numpy as np
import pandas as pd
data = [15, 16, 18, 19, 22, 24, 29, 30, 34]

print("mean: ", np.mean(data))
print("median: ", np.median(data))
print("50th percentile (median): ", np.percentile(data, 50))
print("25th percentile: ", np.percentile(data, 25))
print("75th percentile: ", np.percentile(data, 75))
print("standart deviation: ", np.std(data))
print("variance: ", np.var(data))

df = pd.read_csv('titanic.csv')
print(df.describe())

# converting from DataFrame to numpy array
arrN = df[['Pclass', 'Fare', 'Age']].values
print(arrN)

# determin the size of the array
print('Array size:',arrN.shape)
print('0 row, 1-st column:',arrN[0,1])
print('0 element:',arrN[0])
print('all the rows, 2nd column:',arrN[:,2])

# creating a mask
mask = arrN[:,2] < 18
print('mask:',mask)
print(arrN[mask])

# count of the numbers with True value
print('Count of ppl under 18:', mask.sum())

import matplotlib.pyplot as plt
# scatter
plt.scatter(df['Age'], df['Fare'])
# plt.show()

# What is in a Column
filename = input('write filename: ')
column_name = input('write column name:')

df = pd.read_csv(filename)
print(df[column_name].values)