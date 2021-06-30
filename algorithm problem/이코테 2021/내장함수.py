import math
def lcm(a, b):
    return a*b // math.gcd(a,b)
a = 21
b = 14

print(math.gcd(21, 14))
print(lcm(21, 14))