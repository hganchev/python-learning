print('====== Open Files =======')
file = open("books.text","r")

# Sending "r" means open in read mode, which is the default.
# Sending "w" means write mode, for rewriting the contents of a file.
# Sending "a" means append mode, for adding new content to the end of the file.

# Adding "b" to a mode opens it in binary mode, which is used for non-text files (such as image and sound files).

# open("filename.text","w")
# open("filename.text","r")
# open("filename.text","wb")

print('====== Read Files =======')
# cont = file.read()
# print(cont)

# To read only a certain amount of a file, you can provide the number of bytes to read as an argument to the read function.
# Each ASCII character is 1 byte:

# print(file.read(10))
# for line in file.readlines():
#     print(line)

for line in file:
    print(line)
file.close()

print('====== Writing Files =======')
file = open("books.text","w")
file.write("wheel of time 8")
file.close()

file = open("books.text","a")
file.write("\nwheel of time 9")
file.close()

print('====== Working Files =======')
try:
    f = open("books.text")
    cont = f.read()
    print(cont)
finally:
    f.close()

# An alternative way of doing this is by using with statements.
with open("books.text") as f:
    print(f.read())

