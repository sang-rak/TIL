# 고정되어있을 때는 for 조합이 더 쉽다.
# 4C3
a = [1, 2, 3, 4, 5]
n = len(a)
'''
for i in range(0, n-2):
    for j in range(i+1, n-1):
        for k in range(j+1, n):
            print(a[i],a[j],a[k])
'''

# 3분할 n-1 comb 2
for i in range(0, n-2):
    for j in range(i+1, n-1):
        print(a[:i+1], a[i+1:j+1], a[j+1:n])