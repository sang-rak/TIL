import sys
sys.stdin = open('최장 경로.txt')


def dfs(v, cnt):
    global ans
    visited[v] = 1
    if ans < cnt : ans = cnt

    for w in adj[v]:
        if not visited[w]:
            dfs(w, cnt+1)




T = int(input())

for tc in range(1, T+1):
    V, E = map(int, input().split())
    # 인접리스트
    adj = [[] for _ in range(V+1)]
    visited = [0] * (V+1)
    ans = 0
    for i in range(E):
        s, e = map(int, input().split())
        adj[s].append(e)
        adj[e].append(s)

    for i in range(1, V+1): # 각 노드를 출발점으로
        visited = [0] * (V + 1)
        dfs(i, 1)


    print("#{} {}".format(tc, ans))