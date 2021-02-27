def atoi(a):
    value = 0
    for i in range(len(a)):
        value = value * 10 + a[i]
    return value


a = [1, 2, 3]
b = atoi(a)
print(a, type(a))
print(b, type(b))