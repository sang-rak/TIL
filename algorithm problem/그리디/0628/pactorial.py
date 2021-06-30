data = [1,2,3,4,5,6,7]
def fact(n):
    if n <= 1:
        return 1
    else:
        return n * fact(n-1)

print(fact(3))
