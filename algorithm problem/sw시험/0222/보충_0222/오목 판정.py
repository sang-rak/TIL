# i,j, 이동 방향, 현재까지 연속된 돌의 숫자
def dfs(i, j, dir, cnt):
    global dfs_break

    # 탈출조건, 연속된 돌의 개수가 5 이상이면
    if cnt >= 5:
        # 이 후 추가적인 탐색을 피하기 위해 변경
        dfs_break = 1
        return

    # 범위 벗어나지 않으면
    if 0 <= i + x[dir] < n and 0 <= j + y[dir] < n:

        # 다음 위치에 돌이 놓여 있으면
        if board[i + x[dir]][j + y[dir]] == 'o':
            dfs(i + x[dir], j + y[dir], dir, cnt + 1)

        else:
            return

    else:
        return


T = int(input())

# →, ↘, ↓, ↙
x = [0, 1, 1, 1]
y = [1, 1, 0, -1]

for tc in range(1, 1 + T):

    # 오목판 크기
    n = int(input())

    # 오목판
    board = [list(map(str, input())) for _ in range(n)]

    # 한 번 찾은 후, 추가적인 탐색을 피하기 위한 기준
    dfs_break = 0

    for q in range(n):

        # 이미 5개 이상인 경우 있으면
        if dfs_break == 1:
            break

        for w in range(n):

            # 이미 5개 이상인 경우 있으면
            if dfs_break == 1:
                break

            # 돌이 놓여있는 곳이면
            if board[q][w] == 'o':

                # 방향 탐색
                for e in range(4):

                    # → 방향이면
                    if e == 0:

                        # 현재 위치에서 오른쪽으로 움직일 수 있는 범위가 5 이상인 경우
                        # 5보다 작으면 탐색할 필요 없음
                        if n - w >= 5:
                            # q,w,이동 방향,현재까지 연속해서 놓인 돌의 개수
                            dfs(q, w, e, 1)

                    # ↘ 방향이면
                    elif e == 1:

                        # 현재 위치에서 오른쪽, 아래로 움직일 수 있는 범위가 5 이상인 경우

                        if n - w >= 5 and n - q >= 5:
                            dfs(q, w, e, 1)

                    # ↓ 방향이면
                    elif e == 2:

                        # 현재 위치에서 아래로 움직일 수 있는 범위가 5 이상인 경우
                        if n - q >= 5:
                            dfs(q, w, e, 1)

                    # ↙ 방향이면
                    else:

                        # 현재 위치에서 아래로 움직일 수 있는 범위가 5 이상인 경우
                        # 현재 위치에서 왼쪽으로 움직일 수 있는 범위가 4 이상인 경우
                        # ex) w가 4이면 0,1,2,3,4 최대 5개 가능
                        if n - q >= 5 and w >= 4:
                            dfs(q, w, e, 1)

                    # 이미 5개 이상인 경우 있으면
                    if dfs_break == 1:
                        break

    # 5개 이상인 경우 존재하면
    if dfs_break == 1:
        print('#{} {}'.format(tc, 'YES'))

    # 존재하지 않으면
    else:
        print('#{} {}'.format(tc, 'NO'))