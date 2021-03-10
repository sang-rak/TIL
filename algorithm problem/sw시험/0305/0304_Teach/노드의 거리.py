import sys
sys.stdin = open("노드의 거리.txt","r")


def bfs(sv):

    # 큐를 생성과 동시에 선언
    Q = [[sv, 0]]
    # 방문체크를 위한 배열 선언
    visited = [0] * (V + 1)
    visited[sv] = True

    while Q:
        v, dist = Q.pop(0)

        if v == G:
            return dist

        for i in range(V+1):
            if adj[v][i] == 1 and visited[i] == False:
                Q.append([i, dist+1])
                visited[i] = True
    return 0


T = int(input())
for tc in range(1, T+1):
    V, E = map(int, input().split())



    adj = [[0] * (V+1) for _ in range(V+1)]
    for i in range(E):
        s, e = map(int, input().split())
        adj[s][e] = 1
        adj[e][s] = 1

    S, G = map(int, input().split())

    print("#{} {}".format(tc, bfs(S)))
