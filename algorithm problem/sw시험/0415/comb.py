def comb(n, r):
    if r == 0:
        print(T)
    elif n < r:
        return
    else:
        T[r-1] = A[n-1]
        comb(n-1, r-1)
        comb(n-1, r)


N = 4
R = 3
A = [1,2,3,4]
T = [0] * R
comb(N, R)