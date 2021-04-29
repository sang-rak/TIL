'''
7 8
1 2 1 3 2 4 2 5 4 6 5 6 6 7 3 7
'''

def bfs(s, v):
    Q = [s]
    visited = [0] * (V+1)
    visited[s] = 1
    while Q:
        t = Q.pop(0)
        print(t)
        for i in range(1, V+1):
            if adj[t][i] == 1 and visited[i] == 0:
                Q.append(i)
                visited[i] = visited[t] + 1


V, E = map(int, input().split())
edge = list(map(int, input().split()))
adj = [[0] * (V+1) for _ in range(V+1)]
adjlist = [[] for _ in range(V+1)]

for i in range(E):
    n1 = edge[i*2]
    n2 = edge[i*2+1]
    adj[n1][n2] = 1
    adjlist[n1].append(n2)

bfs(1, V)