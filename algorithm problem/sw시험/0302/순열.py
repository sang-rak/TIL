def perm(n, k):
    if n == k:
        print(order)
    else:
        for i in range(n):
            # True면 넘기기
            if visited[i]: continue
            # 없으면 진행
            order[k] = arr[i]
            visited[i] = True
            perm(n, k+1)
            visited[i] = False


arr = [1,2,3]
N = len(arr)
order = [0] * N
visited = [0] * N
perm(N, 0)