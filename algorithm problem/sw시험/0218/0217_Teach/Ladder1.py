import sys
sys.stdin = open("Ladder1.txt")
T = 10
SIZE = 100
dx = [0, 0, -1] # 좌 우 상
dy = [-1, 1, 0]



# 2인곳 찾기
def find_start(arr):
    for i in range(SIZE):
        if arr[SIZE-1][i] == 2:
            x = SIZE - 1
            y = i
            return x, y

def check(arr ,x ,y):
    if x < 0 or x >= SIZE : return False
    if y < 0 or y >= SIZE : return False
    if arr[x][y] == 0: return False
    if arr[x][y] == 9: return False # 방문표시
    return True


def ladder(arr):  # 아래서 위로 올라감
    x, y = find_start(arr)  # 현재 좌표
    nx, ny = 0, 0           # 다음 좌표
    # 세 방향 확인
    while True:
        if x == 0: return y  # 첫행에 도착
        for i in range(3):   # 3방향으로 검색
            nx = x + dx[i]
            ny = y + dy[i]
            if check(arr, nx, ny): # 이동 가능한 방향 체크
                x, y = nx, ny
                arr[x][y] = 9
                break
for tc in range(1, T+1):
    no = int(input())  # 번호
    arr = [list(map(int, input().split())) for _ in range(SIZE)] # 사다리판

    print("#{} {}".format(tc, ladder(arr)))