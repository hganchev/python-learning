from typing import Sequence
import pandas as pd
import os
import datetime


def WaitForText():
    """ Waiting for user input string
    :input: input string
    :return: textToWrite string
    """
    textToWrite = input("Write text: ")
    answer = input("Are you sure you want to write it? y/n").strip().lower()
    if answer == "y" :
        print("You chose - yes")
        return textToWrite
    else:
        print("You chose - no")
        return ""

# Cycle waiting for input string
textToWrite = ""
while textToWrite == "":
    textToWrite = WaitForText()    
else:
    pass       
print(textToWrite)

# Take all the xls files from Files dir
listTables=[]
index = 0
for file in os.listdir('Files/') :
    if file.endswith(".xls"):
        listTables.insert(index,file)
        index =+ 1
print(listTables)

# Wait for choosing the file number
objectToWrite = input("Choose file number to write: ")
print(objectToWrite)

# Write to file number name
for entry in listTables:
    if objectToWrite in entry:
        path = os.path.join("Files", entry)
        if os.path.exists(path):
            table = pd.read_excel(io=path, engine='xlrd')
            print(table)
            new_row = pd.DataFrame({'Date': [datetime.datetime.now()], 'Text': [textToWrite]})
            table = table.append(new_row, ignore_index=True)
            print(table)
            table.to_excel(path, index=False)
        else:
            print(f"{path} does not exist")
        pass