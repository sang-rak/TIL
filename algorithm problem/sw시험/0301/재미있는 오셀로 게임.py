import sys
sys.stdin = open("재미있는 오셀로 게임.txt","r")
T = int(input())

dy = [-1, 1, 0, 0, 1, -1, 1, -1]
dx = [0, 0, -1, 1, 1, 1, -1, -1]

def lay_check(x, y, color):

    # 행열 맞추기
    x = x - 1
    y = y - 1

    lay_stone(x, y, color)

def check_board(x, y, color):
    other_colors = []
    for i in range(8):

        nx = x + dx[i]
        ny = y + dy[i]

        if 0 <= nx < N and 0 <= ny < N:
            other_color = []
            while 0 <= nx < N and 0 <= ny < N:

                if board[ny][nx] == 0:
                    break
                if board[ny][nx] == color:
                    other_colors.append(other_color)
                    break
                else:
                    other_color.append((nx, ny))

                nx += dx[i]
                ny += dy[i]

    reverse_color(other_colors, color)

def reverse_color(locations, color):
    for location in locations:
        for x, y in location:
            board[y][x] = color


def lay_stone(x, y, color):
    board[y][x] = color
    check_board(x, y, color)


def lay_center():
    board[N//2][N//2], board[N//2 - 1][N//2] = 2, 1
    board[N//2 - 1][N//2 - 1], board[N//2][N//2 - 1] = 2, 1

def win_board(board):
    white = 0
    black = 0
    for i in range(N):
        for j in range(N):
            if board[i][j] == 1:
                white += 1
            elif board[i][j] == 2:
                black += 1

    return white, black

for tc in range(1, T+1):
    N, M = map(int, input().split())

    board = [[0] * N for _ in range(N)]
    lay_center()

    for i in range(M):
        x, y, color = list(map(int, input().split()))
        lay_check(x, y, color)

    white, black = win_board(board)

    print("#{} {} {}".format(tc, white, black))