
def powerset(n, k): # 원소의 개수, k: 뎁스
    if n == k:
        print(bit)
        for i in range(N):
            if bit[i]:
                print(arr[i], end=' ')
        print()
    else:
        bit[k] = 1
        powerset(n, k+1)
        bit[k] = 0
        powerset(n, k+1)

arr = [1, 2, 3]
N = len(arr)
bit = [0] * N

powerset(N, 0)

