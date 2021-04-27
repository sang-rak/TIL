import sys
sys.stdin = open('정사각형 방.txt', 'r')

T = int(input())

for tc in range(1, T+1):
    N = int(input())
    arr = [list(map(int, input().split())) for _ in range(N)]
    v = [[0] * N for _ in range(N)]

    dr = [0, 0, -1, 1]
    dc = [1, -1, 0, 0]

    # 주위에 이동할 곳이 있다면 1로 변환

    for i in range(N):
        for j in range(N):
            while True:
                for k in range(4):
                    # 범위 안에 있다면
                    if 0 <= i+dr[k] < N and 0 <= j+dc[k] < N:
                        # 주변에 a[i][j]+1 있으면
                        if arr[i][j]+ 1 == arr[i+dr[k]][j+dc[k]]:
                            v[i][j] = 1
    print(v)

    # 연속한 숫자가 가장 길고 같으면 가장 적은 방수는 출력

    # 방의 개수가 최대인 방이 여럿이라면 그 중에서 적힌 수가 가장 작은 것을 출력

    # 처음에 출발해야 하는 방 번호와 최대 몇 개의 방을 이동할 수 있는지를 공백으로 구분하여 출력