import sys
sys.stdin = open("미로.txt", "r")

def dfs(x, y):
    global flag
    # 가지치기
    if arr[x][y] == 3:
        flag = 1
        return

    # 방문처리
    arr[x][y] = 9

    # 시작점에 인접한 정점(w)중에 방문안 한 정점이 있으면
    for i in range(4):  # 상하좌우
        nx = x + dx[i]
        ny = y + dy[i]

        # 0은 통로, 1은 벽, 2는 출발, 3은 도착, 9는 방문
        if nx < 0 or nx >= N: continue
        if ny < 0 or ny >= N: continue
        if arr[nx][ny] == 9: continue
        if arr[nx][ny] == 1: continue
        dfs(nx, ny)

def find_start(arr):
    for x in range(N):
        for y in range(N):
            if arr[x][y] == 2:
                return (x, y)


T = int(input())

dx = [-1, 1, 0, 0]
dy = [0, 0, 1, -1]

for tc in range(1, T+1):
    flag = 0  # dfs
    N = int(input())
    arr = [list(map(int, input())) for _ in range(N)]

    # 시작점 찾기
    sx, sy = find_start(arr)
    dfs(sx, sy)
    print("#{} {}".format(tc, flag))
