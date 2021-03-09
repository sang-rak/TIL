import sys
sys.stdin = open('미로의 거리.txt', 'r')

def check_start(adj):
    # 시작 x,y 찾기
    for i in range(N):
        for j in range(N):
            if adj[i][j] == 2:
                return i, j

def inside(y, x):
    return 0 <= y < N and 0 <= x < N and (adj[y][x] == 0 or adj[y][x] == 3)

    # 0은 통로, 1은 벽, 2는 출발, 3은 도착이다.
def bfs(r, c):
    global result

    Q.append((r, c))
    visited.append((r, c))

    # Q가 not empty:
    while Q:
        # t <- deQ, 부모
        r, c = Q.pop(0)

        for move in range(4):
            nr = r + dr[move]
            nc = c + dc[move]
            # w가 not visited 면
            if inside(nr, nc) and (nr, nc) not in visited:
                # endQ
                Q.append((nr, nc))
                visited.append((nr, nc))
                distance[nr][nc] = distance[r][c] + 1
                if adj[nr][nc] == 3:
                    result = distance[nr][nc] - 1
                    return




T = int(input())

for tc in range(1, T+1):

    N = int(input())

    adj = [list(map(int, input())) for _ in range(N)]
    visited = [[0] * N for _ in range(N)]

    # 시작점 구하기

    r, c = check_start(adj)

    dr = [-1, 1, 0, 0]
    dc = [0, 0,-1, 1]

    result = 0
    Q = []
    distance = [[0] * N for _ in range(N)]
    bfs(r, c)
    print("#{} {}".format(tc, result))

