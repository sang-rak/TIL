import sys
sys.stdin = open("노드의 거리.txt","r")

T = int(input())

def bfs(v):
    # v를 enQ(visited 체크)
    Q.append(v)
    visited[v] = 1

    while len(Q) != 0:

        # 부모
        t = Q.pop(0)

        for w in range(1, V+1):
            if adj[t][w] == 1 and visited[w] == 0:
                Q.append(w)
                visited[w] = visited[t] + 1



for tc in range(1, T+1):
    V, E = map(int, input().split())
    adj = [[0] * (V+1) for _ in range(V+1)]

    # 영 행렬에 간선 그리기
    for _ in range(E):
        s, e = map(int, input().split())
        adj[s][e] = 1
        adj[e][s] = 1

    # 방문 로그 남기기
    visited = [0] * (V+1)

    S, G = map(int, input().split())

    Q = []
    bfs(S)
    if visited[G] == 0:
        print('#{} 0'.format(tc))
    else:
        print("#{} {}".format(tc, visited[G]-1))