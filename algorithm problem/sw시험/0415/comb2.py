a = [1,2,3,4]
N = 4
for i in range(0, N-2):
    for j in range(i+1, N-1):
        for k in range(j+1, N):
            print(a[i], a[j], a[k])
print()


def comb(k, s):
    if k == R:
        print(t)
    else:
        for i in range(s, N - R + k + 1):
            t[k] = a[i]
            comb(k+1, i+1)


def H(k, s):
    if k == R:
        print(t)
    else:
        for i in range(s, N):
            t[k] = a[i]
            H(k+1, i)


N = 4
R = 3
a = [1,2,3,4]
t = [0] * R
comb(0,0)
print()
# 중복순열
H(0,0)