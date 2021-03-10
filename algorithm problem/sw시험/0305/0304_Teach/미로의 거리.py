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
    # 선형큐를 이용해서 작성을 해보자
    Q = [0] * 100000
    front = -1
    rear = 0
    Q[rear] = (r,c)

    dist = [[-1] * (N) for _ in range(N)]
    dist[r][c] = 0

    # 선형큐에서의 공백 검사
    while front != rear:
        front += 1
        curr_r, curr_c = Q[front]
        if adj[curr_r][curr_c] == '3':
            return dist[curr_r][curr_c] - 1
        for i in range(4):
            nr = curr_r + dr[i]
            nc = curr_c + dc[i]

            # 벽으로 둘러싸지 않았기 때문에 범위 검사를 해야한다.
            if nr < 0 or nr >= N or nc < 0 or nc >= N: continue

            #벽이 아니면서 거리를 갱신하지 않았다면 좌표를 넣고 갱신
            if adj[nr][nc] != '1' and dist[nr][nc] == -1:
                dist[nr][nc] = dist[curr_r][curr_c] + 1
                rear += 1
                Q[rear] = (nr,nc)
    return 0


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

    print("#{} {}".format(tc, bfs(r, c)))

