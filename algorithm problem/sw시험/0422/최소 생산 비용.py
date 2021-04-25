import sys
sys.stdin = open('최소 생산 비용.txt', 'r')


def perm(n, k, cursum):
    global min
    # 가지치기
    if min <= cursum: return

    if n == k:
        if cursum < min: min = cursum
    else:
        for i in range(n):
            if visited[i]: continue
            visited[i] = 1
            order[k] = i
            perm(n, k+1, cursum + arr[k][i])
            visited[i] = 0

T = int(input())

for tc in range(1, T + 1):
    N = int(input())
    arr = [list(map(int, input().split())) for _ in range(N)]
    order = [0] * N
    visited = [0] * N
    min = 987654321

    perm(N, 0, 0)

    print("#{} {}".format(tc, min))