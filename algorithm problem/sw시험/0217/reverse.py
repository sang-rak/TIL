def my_reverse(s):
    arr = list(s)
    n = len(arr)
    for i in range(n//2):
        arr[i], arr[n-1-i] = arr[n-1-i], arr[i]
    arr = "".join(arr)
    return arr

s = "abcd"
s2 = my_reverse(s)
print(s, type(s))
print(s2, type(s2))

# s = 'abcd'
# s = s[::-1]
# arr = list(s) # str -> list
# arr.reverse()
