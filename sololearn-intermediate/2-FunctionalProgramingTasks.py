# Introduction - style of programming based on functions
# a key part is higher-order functions - functions that take other functions as arguments
# import os
# os.system('cls' if os.name == 'nt' else 'clear')
print('====== Introduction =======')
def apply_twice(func, arg):
    return func(func(arg))

def add_five(x):
    return x + 5

print(apply_twice(add_five,10))

# test
def test(func, arg):
  return func(func(arg))

def mult(x):
  return x * x

print(test(mult, 2))
# Pure functions - returns value that depends only on their arguments
# example cos(x)
print('====== Pure Functions =======')
def pure_function(x, y):
    temp = x + 2*y
    return temp/(2*x + y)

# Pure functions - changes the state of the argument
print('====== Impure Functions =======')
some_list = []

def impure(arg):
    some_list.append(arg)

# Lambda functions - Python allows to create a function on-the-fly that
#  provided that they are created using lambda syntax, known as anonymous
print('====== Lambda Functions =======')

def my_func(f,arg):
    return f(arg)

print(my_func(lambda x: 2*x*x, 5))

# The built-in functions map and filter are very useful higher-order functions that operate on lists (or similar objects called iterables).
# The function map takes a function and an iterable as arguments, and returns a new iterable with the function applied to each argument
print('====== Map,filter =======')

nums = [11,22,33,44,55]
result = list(map(add_five, nums))
print(result)
result = list(map(lambda x: x+5, nums))
print(result)
result = list(filter(lambda x: x%2==0, nums))
print(result)

# Generators - are a type of iterable, like lists or tuples.
# Unlike lists, they don't allow indexing with arbitrary indices, but they can still be iterated through with for loops.
# They can be created using functions and the yield statement.
# !The yield statement is used to define a generator, replacing the return of a function to provide a result to its caller without destroying local variables.
print('====== Generators =======')
def countdown():
    i = 5
    while i > 0:
        yield i
        i -= 1

for i in countdown():
    print(i)

# Due to the fact that they yield one item at a time, generators don't have the memory restrictions of lists.
# In fact, they can be infinite!

# def infinite_sevens():
#     while True:
#         yield 7

# for i in infinite_sevens():
#     print(i)

# Finite generators can be converted into lists by passing them as arguments to the list function.
def numbers(x):
    for i in range(x):
        if i%2 == 0:
            yield i

print(list(numbers(11)))
# Decorators -Decorators provide a way to modify functions using other functions.
# This is ideal when you need to extend the functionality of functions that you don't want to modify.
print('====== Decorators =======')
def decor(func):
    def wrap():
        print('==============')
        func()
        print('==============')
    return wrap

@decor # second method
def print_text():
    print("Hello Hristo")

# first method to do decorators
# decorated = decor(print_text)
# decorated()
print_text()

# Recursion - Recursion is a very important concept in functional programming.
# The fundamental part of recursion is self-reference -- functions calling themselves. 
# It is used to solve problems that can be broken up into easier sub-problems of the same type.
# ! The base case acts as the exit condition of the recursion.
# Not adding a base case results in infinite function calls, crashing the program.
print('====== Reccursion =======')
def factorial(x):
    if x==1:
        return 1
    else:
        return x * factorial(x-1)

print(factorial(5))

def is_even(x):
    if x == 0:
        return True
    else:
        return is_odd(x-1)

def is_odd(x):
    return not is_even(x)

print(is_odd(17))
print(is_even(23))

# Test 
def fib(x):
  if x == 0 or x == 1:
    return 1
  else: 
    return fib(x-1) + fib(x-2)
print(fib(4))

print('====== **args and **kwargs =======')

# *args
# Python allows you to have functions with varying numbers of arguments.
# Using *args as a function parameter enables you to pass an arbitrary number of arguments to that function.
#  The arguments are then accessible as the tuple args in the body of the function
# retruns a tuple
print('====== **args =======')
def function(named_arg,*args):
    print(named_arg)
    print(args)

function(1,2,3,4,5)

# **kwargs
# **kwargs (standing for keyword arguments) allows you to handle named arguments that you have not defined in advance.
# The keyword arguments return a dictionary in which the keys are the argument names, and the values are the argument values.
print('====== **kwargs =======')
def my_func(x,y=7,*args,**kwargs):
    print(kwargs)

my_func(2,3,4,5,6,a=7,b=8)

# Quiz
nums = {1, 2, 3, 4, 5, 6}
nums = {0, 1, 2, 3} & nums
nums = filter(lambda x: x > 1, nums)
print(len(list(nums)))

def power(x, y):
  if y == 0:
    return 1
  else:
    return x * power(x, y-1)
		
print(power(2, 3))