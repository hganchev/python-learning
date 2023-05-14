# Description
# Given a string as input, use recursion to output each letter of the strings in reverse order, on a new line.

def spell(txt):
    if txt == "":
        return txt
    else:
        return spell(txt[1:]) + txt[0]

txt = "HELLO"
print(spell(txt))
