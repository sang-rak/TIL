import sys
sys.stdin = open("배열2_대각 최대합.txt", "r")

T = int(input())


def check(arr):

    # 좌상 좌하 우상 우하
    dr = [-1, 1, -1, 1]
    dc = [-1, -1, 1, 1]

    max_cnt = 0

    for i in range(N):
        for j in range(N):
            # 좌표 값
            cnt = arr[i][j]

            for k in range(4):
                dx = i + dr[k]
                dy = j + dc[k]

                # board 내에 있으면
                while 0 <= dx < N and 0 <= dy < N:

                    # 이동 후 값 더하기
                    cnt += arr[dx][dy]
                    dx = dx + dr[k]
                    dy = dy + dc[k]

            # max 값 갱신
            if max_cnt < cnt:
                max_cnt = cnt

    return max_cnt

for tc in range(1, T+1):

    N = int(input())
    arr = [list(map(int, input().split())) for _ in range(N)]

    result = check(arr)
    print('#{} {}'.format(tc, result))