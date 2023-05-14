import numpy as np
# σ - population standard deviation
# µ - mean
# One standard deviation (µ ± σ)
# Two standard deviations (µ ± 2σ)
# Three standard deviations (µ ± 3σ)

# problem statement: 
# Calculate and output(print) the percentage of houses 
# that are within one standard deviation from the mean
# *hint To calculate percentage divide the number of hauses that satisfy the condition
# by the total number of houses, and multiply the result by 100

data = np.array([150000, 125000, 320000, 54000, 200000, 160000, 230000, 280000, 290000,
300000, 500000, 420000, 100000, 150000, 280000])

total_houses = len(data)
print("total houses = " + str(total_houses))

min = np.mean(data) - np.std(data)
min = round(min)
max = np.mean(data) + np.std(data)
max = round(max)

print("min = " + str(min) + "\nmax = " + str(max))
houses_within = []
for house in data:
    # if house >= min and house <= max:
    if house in range(min,max):
        houses_within.append(house)

house_percentage = (len(houses_within)/total_houses) * 100
print(house_percentage)
