'''
#메모이제이션
def fibo1(n):
    if n >= 2 and len(memo) <= n:
        memo.append(fibo1(n-1)+fibo1(n-2))
    return memo[n]
memo = [0, 1]

print(fibo1(40))
'''
'''
# 두번째 방법
memo2 = [-1] * 21
memo2[0] = 0
memo2[1] = 1

def fibo2(n):
    if memo2[n] == -1:
        memo2[n] = fibo2(n-1) + fibo2(n-2)

    return memo2[n]
print(fibo2(40))
print(memo2)
'''

def fibo2(n):
    f = [0, 1]
    for i in range(2, n+1):
        f.append(f[i-1] + f[i-2])

    return f[n]

print(fibo2(1000))