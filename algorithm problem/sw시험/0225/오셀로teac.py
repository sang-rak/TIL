dy = [1, -1, 0, 0, 1, -1, 1, -1]
dx = [0, 0, 1, -1, -1, 1, 1, -1]


def check_board():
    for y in range(n):
        for x in range(n):
            if board[y][x] == 0:
                return True
    return False


def lay_center():
    board[n // 2 - 1][n // 2 - 1] = 2
    board[n // 2 - 1][n // 2] = 1
    board[n // 2][n // 2 - 1] = 1
    board[n // 2][n // 2] = 2


def check_location(y, x):
    return 0 <= y < n and 0 <= x < n


def lay_stone(y, x, color):
    board[y][x] = color
    check_stone(y, x, color)


def check_stone(y, x, color):
    other_colors = []
    for i in range(8):
        j = 1
        ny = y + dy[i] * j
        nx = x + dx[i] * j
        other_color = []
        while True:
            if not check_location(ny, nx):
                break
            if board[ny][nx] == 0:
                break
            elif board[ny][nx] == color:
                other_colors.append(other_color)
                break
            else:
                other_color.append((ny, nx))

            j += 1
            ny = y + dy[i] * j
            nx = x + dx[i] * j
    reverse_stone(other_colors, color)


def reverse_stone(locations, color):
    for location in locations:
        for y, x in location:
            board[y][x] = color


T = int(input())
for t in range(T):
    n, m = list(map(int, input().split()))
    board = [[0] * n for _ in range(n)]
    stones = []
    lay_center()

    for _ in range(m):
        y, x, color = list(map(int, input().split()))
        y = y - 1
        x = x - 1
        stones.append((y, x, color))

    for y, x, color in stones:
        if check_board():
            lay_stone(y, x, color)
        else:
            break

    white = 0
    black = 0

    for y in range(n):
        for x in range(n):
            if board[y][x] == 1:
                black += 1
            elif board[y][x] == 2:
                white += 1
    print("#{} {} {}".format(t + 1, black, white))