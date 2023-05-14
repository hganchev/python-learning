# description
# Create a program to take a string as input and output(print) a dictionary,
#  which represents the letter count
text = 'hello'
dict = {}

for key in text:
    dict[key] = text.count(key)
print(dict)