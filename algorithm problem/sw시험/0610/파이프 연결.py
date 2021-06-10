import sys
sys.stdin = open('파이프 연결.txt', 'r')

T = int(input())

for tc in range(1, T+1):
    N = int(input())
    arr = []
    movearr = [[0] * N for i in range(N)]

    # 현재 위치
    x = 0
    y = 0
    # 움직인 횟수
    cnt = 0

    # 행렬 변환
    for i in range(N):
        arr.append(list(map(int, input().split())))


    # 1은 현재 위치이동
    while x < N and y < N:

        # 실패
        if arr[x][y] == 0:
            break

        # 도착
        elif x == N and y == N:
            print('도착')

        elif arr[x][y] == 1 and (arr[x][y+1] == 1 or arr[x][y+1] == 4 or arr[x][y+1] == 5):  # 왼쪽에서 들어왔을 때 오른쪽 이동
            y += 1
            cnt += 1
            movearr[x][y] = cnt

        elif arr[x][y] == 2 and (arr[x+1][y] == 2 or arr[x+1][y] == 5 or arr[x+1][y] == 6): # 위에서 들어왔을 때 아래 이동
            x += 1
            cnt += 1
            movearr[x][y] = cnt

        elif arr[x][y] == 3 and (arr[x][y+1] == 1 or arr[x][y+1] == 4 or arr[x][y+1] == 5): # 밑에서 들어왔을때 오른쪽 이동
            y += 1
            cnt += 1
            movearr[x][y] = cnt

        elif arr[x][y] == 4 and (arr[x+1][y] == 2 or arr[x+1][y] == 5 or arr[x+1][y] == 6):  # 왼쪽에서 들어왔을 때 아래 이동
            x += 1
            cnt += 1
            movearr[x][y] = cnt

        elif arr[x][y] == 5 and (arr[x-1][y] == 2 or arr[x-1][y] == 3):  # 왼쪽에서 들어왔을 때 위쪽 이동
            x += -1
            cnt += 1
            movearr[x][y] = cnt

        elif arr[x][y] == 6 and (arr[x][y+1] == 1 or arr[x][y+1] == 4 or arr[x][y+1] == 5): # 위쪽에서 들어왔을 때 오른쪽 이동
            y += 1
            cnt += 1
            movearr[x][y] = cnt

        else:
            break


    print(movearr)