import matplotlib.pyplot as plt
import pandas as pd

s = pd.Series([18,42,9,32,81,64,3])

s.plot(kind='bar')
# plt.savefig('plot.png')
plt.xlabel('x values')
plt.ylabel('y values')
plt.show()

# bar plots
# hist = histogram plots
# area - stacked as default
# scatter - gives the reletion between two variables
# pie - pie chart
# subtitle - set plot title

data = {'sport':["Soccer","Tennis", "Soccer","Hockey"],
'players':[5,4,8,20]}

df = pd.DataFrame(data)
df.groupby('sport')["players"].sum().plot(kind='pie')
plt.show()