import sys
sys.stdin = open('배열 최소 합.txt', 'r')


def dfs(idx, sum_result):
    global min_result
    if idx == N:
        if sum_result < min_result:
            min_result = sum_result
        return

    # 가지치기
    if sum_result >= min_result:
        return
    for i in range(N):


        if visited[i]:
            visited[i] = False
            dfs(idx + 1, sum_result + arr[idx][i])
            visited[i] = True


T = int(input())
for tc in range(1, T + 1):
    N = int(input())
    arr = [list(map(int, input().split())) for _ in range(N)]
    visited = [True for _ in range(N)]
    min_result = 10^100
    dfs(0, 0)
    print("#{} {}".format(tc, min_result))