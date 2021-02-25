import sys

sys.stdin = open("Ladder1.txt", "r")

T = 10

for test_case in range(1, T + 1):
    number = int(input())
    arr = [list(map(int, input().split())) for _ in range(100)]

    #거꾸로 간다. 도착부터
    for i in range(100):
        if arr[99][i] == 2:
            # 도착 x지점 지정
            x = i
            break # 찾으면 멈춤
    # 도착 y지점 지정
    y = 99

    while True:
        # 왼쪽으로 갈 수 있는 경우 왼쪽으로 이동 후 위로
        if x > 0 and arr[y][x-1]:
            # 왼쪽으로 갈 수있는 한계까지 이동
            while x > 0 and arr[y][x-1]:
                x -= 1

            else:
                y -= 1 # 한계점 일 때 위로 감
        # 오른쪽을 갈 수 있는 경우 오른쪽으로 이동 후 위로
        elif x < 99 and arr[y][x+1]:
            # 오른쪽으로 갈 수 있는 한계까지 이동
            while x < 99 and arr[y][x+1]:
                x += 1
            else:
                y -= 1 # 한계점 일때 위로 감
        else: # 왼쪽오른쪽 아무곳도 못간다면 위로
            y -= 1
        if y == 0: # 도착
            break

    print('#{} {}'.format(test_case,x))


