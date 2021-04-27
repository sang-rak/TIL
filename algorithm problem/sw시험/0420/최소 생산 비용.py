import sys
sys.stdin = open('최소 생산 비용.txt', 'r')


def dfs(num, cnt):
    global result

    # 최단거리
    if num == N:
        if result > cnt:
            result = cnt
        return

    # 가지치기
    if result < cnt:
        return

    for i in range(N):
        if not visited[i]:
            visited[i] = True
            dfs(num+1, cnt + data[num][i])
            visited[i] = False

T = int(input())

for tc in range(1, T + 1):
    N = int(input())
    data = [list(map(int, input().split())) for _ in range(N)]
    visited = [0] * N
    result = 999999999
    cnt = 0

    dfs(0, 0)

    print("#{} {}".format(tc, result))