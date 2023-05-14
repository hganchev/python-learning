# You have already seen exceptions in previous code. They occur when something goes wrong, 
# due to incorrect code or input. When an exception occurs, the program immediately stops.
# The following code produces the ZeroDivisionError exception by trying to divide 7 by 0:

num1 = 7
num2 = 0

# print(num1/num2)

# Common exceptions:
# ImportError: an import fails;
# IndexError: a list is indexed with an out-of-range number;
# NameError: an unknown variable is used;
# SyntaxError: the code can't be parsed properly;
# TypeError: a function is called on a value of an inappropriate type;
# ValueError: a function is called on a value of the correct type, but with an inappropriate value.

print('====== Exception Handling =======')

# To handle exceptions, and to call code when an exception occurs, you can use a try/except statement

try:
    num1 = 7
    num2 = 0

    print(num1/num2)  
    print("Done")
except ZeroDivisionError:
    print("An error occured \ndue to zero devision")

# A try statement can have multiple different except blocks to handle different exceptions.

try:
    variable = 10
    print(variable + 'hello')
    print(variable / 2)
except ZeroDivisionError:
    print("Devided by zero")
except (ValueError, TypeError):
    print("Error occured")

print('====== Exception Handling - finaly =======')
# After a try/except statement, a finally block can follow. It will execute after the try/except block, no matter if an exception occurred or not.

try:
    print('hello')
    print(1 / 0)
except ZeroDivisionError:
    print("Devided by zero")
finally:
    print("This will run no matter what")

print('====== Exception Handling - else =======')

# The else statement can also be used with try/except statements.
# In this case, the code within it is only executed if no error occurs in the try statement.

try:
    print('1')
except ZeroDivisionError:
    print("2")
else:
    print("3")

try:
    print(1/0)
except ZeroDivisionError:
    print("2")
else:
    print("3")

print('====== Raising exceptions =======')
num = 102
if num > 100:
    raise ValueError