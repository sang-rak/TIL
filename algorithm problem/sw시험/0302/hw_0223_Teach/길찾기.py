import sys
sys.stdin = open("길찾기.txt")

T = 10
SIZE = 100
S = 0
GOAL = 99

def dfs(v):
    global flag
    if v == GOAL:
        flag = 1
        return

    visited[v] = 1
    for w in range(0, SIZE):
        if adj[v][w] and not visited[w]:
            dfs(w)

for tc in range(1, T+1):
    flag = 0
    no, E = map(int, input().split())
    temp = list(map(int, input().split())) # 간선들의 정보

    # 인접행렬
    adj = [[0] * SIZE for _ in range(SIZE)]

    # 방문 체크
    visited = [0] * SIZE

    for i in range(0, len(temp), 2):
        adj[temp[i]][temp[i+1]] = 1  # 방향성 있음
    dfs(S)

    print("#{} {}".format(tc, flag))
