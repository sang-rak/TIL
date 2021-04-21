'''
7 8
1 2 1 3 2 4 2 5 4 6 5 6 6 7 3 7
'''
def dfs(v):
    visited[v] = 1
    print(v, end= " ")
    # for w in range(1, V+1):
    #     if G[v][w] == 1 and visited[w] == 0
    for w in G[v]:
        if visited[w] == 0:
            dfs(w)

def bfs(v):
    #enQueue
    Q.append(v)
    visited[v] = 1
    while Q:
        #deQueue
        v = Q.pop(0)
        print(v, end=" ")
        for w in G[v]:
            if not visited[w]:
                Q.append(w)
                visited[w] = visited[v] + 1

V, E = map(int, input().split())
temp = list(map(int,input().split()))
#인접리스트 7 X 1
# G = [[] for _ in rane(V+1)]
G = {i:[] for i in range(1, V+1)}
for i in range(E):
    s, e = temp[2*i], temp[2*i+1]
    G[s].append(e)
    G[e].append(s)

visited = [0] *(V+1)
dfs(1)
visited = [0] *(V+1)
Q = []
bfs(1)
