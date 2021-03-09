import sys
import time
before = time.time()
sys.stdin = open('배열 최소 합.txt', 'r')

def perm(n, k, cursum):
    global ans

    if ans < cursum:
        return

    if n == k:
        if ans > cursum: ans = cursum
    else:
        for i in range(N):
            if visited[i]: continue
            order[k] = arr[k][i]
            visited[i] = 1
            perm(n, k+1, cursum + order[k])
            visited[i] = 0


T = int(input())
for tc in range(1, T + 1):
    ans = 987654321
    N = int(input())
    arr = [list(map(int, input().split())) for _ in range(N)]
    order = [0] * N
    visited = [0] * N
    perm(N, 0, 0)
    print("#{} {}".format(tc, ans))

print(time.time() -before)