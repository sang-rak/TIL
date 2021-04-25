




def bfs(s, V):
    Q = [s]
    visited = [0] * (V+1)
    visited[s] = 1
    while Q:
        t = Q.pop(0)
        print(t)
        for i in range(1, V+1):
            if adj[t][j] == 1 and visited[i] == 0
                Q.append(i)
                visited[i] = visited[t] + 1
