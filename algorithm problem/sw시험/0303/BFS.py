'''
7 8
1 2 1 3 2 4 2 5 4 6 5 6 6 7 3 7
'''

def bfs(v):
    # v를 enQ(visited체크)
    Q.append(v)
    visited[v] = 1
    print(v, end=" ")

    # Q가 not empty:
    while len(Q) != 0:

        # t <- deQ, 부모
        t = Q.pop(0)
        # t에 인접한 모드 정점 w에 대해서, w: 자식
        for w in range(1, V+1):
            # w가 not visited 면
            if adj[t][w] == 1 and visited[w] == 0:
                # endQ
                Q.append(w)
                visited[w] = visited[t] + 1
                print(w, end=" ")

V, E = map(int, input().split())  # 정점, 간선
temp = list(map(int, input().split()))
adj = [[0] * (V+1) for _ in range(V+1)]  # 인접행렬
Q = []
visited = [0] * (V+1)
# 인접행렬 입력
for i in range(E): # 간선의 수만큼 반복
    s, e = temp[2*i], temp[2*i+1] # 한쌍씩 가져오기
    # 무향 그래프
    adj[s][e] = 1
    adj[e][s] = 1

bfs(1)
max_idx = 0
for i in range(1, V+1):
    if visited[i] > visited[max_idx]:
        max_idx = i
print()
print(max_idx, visited[max_idx]-1)