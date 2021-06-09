import sys
sys.stdin = open('칩 생산.txt', 'r')

T = int(input())

for tc in range(1,T+1):
    H, W = map(int, input().split())
    arr = []
    # 행렬 만들기
    for i in range(H):
        arr.append(list(map(int, input().split())))

    count = 0
    # 계산
    for i in range(H):
        for j in range(W):
            if i+1 < H and j+1 < W:
                if arr[i][j] == 0 and arr[i+1][j] == 0 and arr[i][j+1] == 0 and arr[i+1][j+1] == 0:
                    arr[i][j] = 1
                    arr[i+1][j] = 1
                    arr[i][j+1] = 1
                    arr[i+1][j+1] = 1
                    count += 1
    print("#{} {}".format(tc, count))
    print("#", end='')
    print(tc, end=' ')
    print(count)