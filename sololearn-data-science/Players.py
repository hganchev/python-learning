import numpy as np
import statistics as st
# σ - population standard deviation
# µ - mean
# One standard deviation (µ ± σ)
# Two standard deviations (µ ± 2σ)
# Three standard deviations (µ ± 3σ)

# problem statement: 
# You need to calculate how many(count) players are 
# in range of one standard deviation from the mean
# Numpy
players = [180, 172, 178, 185, 190, 195, 192, 200, 210, 190]
min = np.mean(players) - np.std(players)
min = round(min,2)
max = np.mean(players) + np.std(players)
max = round(max,2)
print('min = ' + str(min) + ' max =' + str(max))
players_pass = []
for player in players:
    if player >= min and player <= max:
        players_pass.append(player)

print(len(players_pass)) # Count of players passed
# Statistics 
min_st = st.mean(players) - st.stdev(players)
min_st = round(min_st)
max_st = st.mean(players) + st.stdev(players)
max_st = round(max_st)
print('minst = ' + str(min_st) + ' maxst =' + str(max_st))
players_pass_st = []
for player in players:
    if player >= min_st and player <= max_st:
        players_pass_st.append(player)

print(players_pass_st)

# rule standard deviation limits
# (max - min) / 4 - this is the rule calc for standard dev