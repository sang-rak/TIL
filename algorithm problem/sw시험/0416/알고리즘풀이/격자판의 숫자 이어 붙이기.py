####### visited 이용 ########
def dfs(x, y, k, num):
    global cnt
    if k == 7:
        if visit[num] != tc:
            cnt += 1
            visit[num] = tc
        return

    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
        if nx < 0 or nx >= 4 or ny < 0 or ny >=  4: continue
        dfs(nx, ny, k + 1, num * 10 + arr[nx][ny])

import sys
sys.stdin = open("격자판의 숫자 이어 붙이기.txt")
T = int(input())
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]
visit = [0] * 10000000  # 0 ~ 1111111

for tc in range(1, T+1):
    #visit = [0] * 10000000  # 0 ~ 1111111
    cnt = 0
    arr = [list(map(int, input().split())) for _ in range(4)]

    for i in range(4):
        for j in range(4):
            dfs(i, j, 1, arr[i][j])

    print("#{} {}".format(tc, cnt))
