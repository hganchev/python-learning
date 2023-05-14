# The focal point of Object Oriented Programming (OOP) are objects, which are created using classes.
# classes - describes what the object will be, but is separate from the object itself. In other words, 
# a class can be described as an object's blueprint, description, or definition.
print('====== Classes =======')

# Classes are created using the keyword class and an indented block, which contains class methods (which are functions).

class Cat:
    def __init__(self,color,legs):
        self.color = color
        self.legs = legs

felix = Cat("ginger", 4)
rover = Cat("dog-colored", 4)
stumpy = Cat("brown", 3)

# The __init__ method is the most important method in a class.
# This is called when an instance (object) of the class is created, using the class name as a function.
# All methods must have self as their first parameter, 
# although it isn't explicitly passed, Python adds the 
# self argument to the list for you; you do not need to 
# include it when you call the methods. Within a method definition, self refers to the instance calling the method.

print(felix.color)

print('====== Methods =======')

class Dog:
    def __init__(self,name,color):
        self.color = color
        self.name = name
    def bark(self):
        print("Woof!")

fido = Dog("Fido", "brown")
print(fido.name)
fido.bark()

print('====== Inherritance =======')
class Animal:
    def __init__(self, name, color):
        self.name = name
        self.color = color
    
class Cat(Animal):
    def purr(self):
        print('Purr...')

class Dog(Animal):
    def bark(self):
        print("Whoof!")

# The function super is a useful inheritance-related function that refers to the parent class.
#  It can be used to find the method with a certain name in an object's superclass.?

class A:
    def spam(self):
        print(1)
    
class B:
    def spam(self):
        print(2)
        # super().spam()

B().spam()

print('====== Magic Methods =======')

# Magic methods are special methods which have double underscores at the beginning and end of their names.
# They are also known as dunders

class Vector2D:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)

first = Vector2D(5,7)
second = Vector2D(3,9)

result = first + second
print(result.x)
print(result.y)

# __sub__ for -
# __mul__ for *
# __truediv__ for /
# __floordiv__ for //
# __mod__ for %
# __pow__ for **
# __and__ for &
# __xor__ for ^
# __or__ for |

# Comparison
# __lt__ for <
# __le__ for <=
# __eq__ for ==
# __ne__ for !=
# __gt__ for >
# __ge__ for >=

# There are several magic methods for making classes act like containers.
# __len__ for len()
# __getitem__ for indexing
# __setitem__ for assigning to indexed values
# __delitem__ for deleting indexed values
# __iter__ for iteration over objects (e.g., in for loops)
# __contains__ for in

print('====== Data Hiding =======')

# A key part of object-oriented programming is encapsulation,
#  which involves packaging of related variables and functions into a single easy-to-use object -- an instance of a class.
# A related concept is data hiding, which states that implementation details of a class should be hidden,
#  and a clean standard interface be presented for those who want to use the class.

# Weakly private methods and attributes have a single underscore at the beginning.
# This signals that they are private, and shouldn't be used by external code.

class Queue:
    def __init__(self,contents):
        self._hiddenlist = list(contents)

    def push(self,value):
        self._hiddenlist.insert(0, value)
    
    def pop(self):
        return self._hiddenlist.pop(-1)

    def __repr__(self):
        return "Queue({})".format(self._hiddenlist)


queue = Queue([1,2,3])
print(queue)
queue.push(0)
print(queue)
queue.pop()
print(queue)
print(queue._hiddenlist)

# Strongly private methods and attributes have a double underscore at the beginning of their names
class Spam:
    __egg = 7
    def print_egg(self):
        print(self.__egg)

s = Spam()
s.print_egg()
print(s._Spam__egg)
# print(s.__egg) #error the __egg is privite and can't be accessed

print('====== Class Methods =======')

# Class methods are different -- they are called by a class, which is passed to the cls parameter of the method.
# Class methods are marked with a classmethod decorator.

# A common use of these are factory methods, which instantiate an instance of a class, using different parameters than those usually passed to the class constructor.

class Rectanle:
    def __init__(self,width, height):
        self.width = width
        self.height = height

    def calculate_area(self):
        return self.width * self.height

    @classmethod
    def new_square(cls, side_length):
        return cls(side_length, side_length)

square = Rectanle.new_square(5)

print(square.calculate_area())

print('====== Static Methods =======')
# Static methods are similar to class methods, except they don't receive any additional arguments; they are identical to normal functions that belong to a class.
# They are marked with the staticmethod decorator.
# ! It's like normal method but can be called from instance of the Class

class Pizza:
    def __init__(self, toppings):
        self.toppings = toppings

    @staticmethod
    def validate_toppins(topping):
        if topping == "pineapple":
            raise ValueError("No pineapples!")
        else:
            return True

ingredients = ["cheese", "onions", "spam"]
if all(Pizza.validate_toppins(i) for i in ingredients):
    pizza = Pizza(ingredients)

print('====== Properties =======')
# Properties provide a way of customizing access to instance attributes.
# They are created by putting the property decorator above a method,
#  which means when the instance attribute with the same name as the method is accessed, the method will be called instead.
# One common use of a property is to make an attribute read-only.


class Pizza2:
    def __init__(self, toppings):
        self.toppings = toppings

    @property
    def pineapple_allowed(self):
        return False

pizza2 = Pizza2(["cheese", "tomato"])
print(pizza2.pineapple_allowed)
# pizza.pineapple_allowed = True #error - read only

# The setter function sets the corresponding property's value.
# The getter gets the value.

class Pizza3:
    def __init__(self, toppings):
        self.toppings = toppings
        self._pineapple_allowed = False

    @property
    def pineapple_allowed(self):
        return self._pineapple_allowed

    @pineapple_allowed.setter
    def pineapple_allowed(self,value):
        if value:
            password = input("Enter the password: ")
            if password == "Sw0rdf1sh":
                self._pineapple_allowed = value
            else:
                raise ValueError("Allert! Intruder!")

pizza3 = Pizza3(["cheese", "tomato"])
print(pizza3.pineapple_allowed)
pizza3.pineapple_allowed = True
print(pizza3.pineapple_allowed)