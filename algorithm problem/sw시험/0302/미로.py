import sys
sys.stdin = open("미로.txt", "r")


def check_load(r, c):
    global flag
    arr[r][c] = 1

    for i in range(4):
        nr = r + dr[i]
        nc = c + dc[i]

        if 0 <= nr < N and 0 <= nc < N:

            if arr[nr][nc] == 3:
                flag = 1
                return

            if arr[nr][nc] == 0:
                check_load(nr, nc)

            nr += dr[i]
            nc += dc[i]


def check_start(arr):
    for i in range(N):
        for j in range(N):
            if arr[i][j] == 2:
                return i, j


T = int(input())

for tc in range(1, T+1):
    N = int(input())

    arr = [list(map(int, input())) for _ in range(N)]

    dr = [-1, 1, 0, 0]
    dc = [0, 0, 1, -1]

    r, c = check_start(arr)

    flag = 0
    check_load(r, c)
    print('#{} {}'.format(tc, flag))