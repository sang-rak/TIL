import sys
sys.stdin = open('오목 판정.txt')
# 아래 위 오른쪽 왼쪽 대각
dr = [1, -1, 0, 0, -1, -1, 1, 1]
dc = [0, 0, 1, -1, -1, 1, -1, 1]

def check():
    # 모든 돌의 위치에서 8방향 탐색을 하겠다.
    for r in range(N):
        for c in range(N):
            # 돌이 아닌경우 건너뛴다.
            if arr[r][c] == '.': continue

            # r,c 돌에서 시작
            for i in range(8):
                nr = r + dr[i]
                nc = c + dc[i]
                cnt = 1
                # 새로운 인덱스를 생성하면 조사
                while 0 <= nr < N and 0 <= nc < N and arr[nr][nc] == 'o':
                    cnt += 1
                    nr = nr + dr[i]
                    nc = nc + dc[i]

                if cnt >= 5:
                    return 1
    return 0

T = int(input())
for tc in range(1, T + 1):
    N = int(input())
    arr = [input() for _ in range(N)]

    ret = check()

    if ret: print('#{} YES'.format(tc))
    else: print('#{} NO'.format(tc))