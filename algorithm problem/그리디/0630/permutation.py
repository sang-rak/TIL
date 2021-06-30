def perm(n, k):
    if k == n:
        print('')
    else:
        for i in range(k,n-1):
            swap