
x = ['hi', 'hello', 'welcome']
print(x[2])
# Dictionaries contains keys and values {key:value} - mutable(can be edited)
ages = {
    "Dave":24,
    "Mary":42,
    "John":58
}

print(ages["Dave"])
print(ages["Mary"])

# Dictionary functions
print('====Dictionaries======')
# in, not in
nums = {
    1:'one',
    2:'two',
    3:'three',
}

print(1 in nums)
print('three' in nums)
print(4 not in nums)

# get
pairs= {
    1:'apple',
    "orange":[2,3,4],
    True: False,
    12:"True"
}

print(pairs.get("orange"))
print(pairs.get(7,42))
print(pairs.get(12345,"not found"))

# Tupples - immutable - cannot be changed
print('====Tupples======')
words = ("spam", "eggs", "sausages")
print(words[0])

# words[1] = "cheese" # gives an error

my_tupple = "one", "two", "three"
print(my_tupple[0])

# tupples unpacking
print('====Tupples unpacking======')
numbers = (1,2,3)
a,b,c = numbers
print(a,b,c)

a,b,*c, d = range(10) #[1,2,3,4,5,6,7,8,9]
print(a,b,c,d)

# Sets
print('====Sets======')
num_set = {1,2,3,4,5}
print(3 in num_set)

# add, remove
nums = {1,2,1,3,1,4,5,6}
print(nums)
nums.add(-7)
nums.remove(3)
print(nums)

# combine two sets
print('====Combine two sets======')
first = {1,2,3,4,5,6}
second = {4,5,6,7,8,9}

print(first|second) # union
print(first&second) # intersection
print(first-second) # difference
print(second-first) # difference
print(first^second) # symmetric difference

# list comperhensions
print('======List Comperhensions =======')
cubes = [i**3 for i in range(5)]
print(cubes)
nums = [i*2 for i in range(10)]
print(nums)
evens = [i**2 for i in range(10) if i**2 % 2 == 0]
print(evens)

# questions
nums = (55, 44, 33, 22)
print(max(min(nums[:2]), abs(-42)))