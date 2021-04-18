########## set() 이용 ##########
def dfs(x, y, k, num):
    if k == 7:
        t.add(num)   #set으로 중복제거
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

for tc in range(1, T+1):
    t = set()
    arr = [list(map(int, input().split())) for _ in range(4)]

    for i in range(4):
        for j in range(4):
            dfs(i, j, 1, arr[i][j])

    print("#{} {}".format(tc, len(t)))
