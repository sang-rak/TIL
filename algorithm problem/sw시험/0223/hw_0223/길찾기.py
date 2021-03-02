
import sys
sys.stdin = open("길찾기.txt")

def dfs(v): # v: 시작정점
    # visited 체크: 하고픈 일 해라(출력)
    visited[v] = True
    # print(v, end=" ")
    # 시작정점(v)의 인접한 모든 점정 (w) for 돌리기
        # 인접정점(w)가 방문하지 않았으면
    for w in range(1, V+1):
        if adj[v][w] == 1 and visited[w] == 0:
            # 다시 dfs(w) 재귀 호출
            dfs(w)

for _ in range(10):

    V = 100
    tc, E = map(int, input().split()) # V: 정점재수, E:간선갯수
    tmp = list(map(int, input().split()))
    adj = [[0]* (V+1) for _ in range(V+1)] # 인접행렬 초기화
    visited = [0] * (V+1)

    # 입력
    for i in range(E):
        s, e = tmp[2*i], tmp[2*i+1]
        adj[s][e] = 1

    # for i in range(V+1):
    #     print("{} {}".format(i, adj[i]))

    print("#{}".format(tc),end=' ')
    dfs(0)
    if visited[99] == True:
        print(1)
    else:
        print(0)

#1 1
#2 1
#3 1
#4 0
#5 1
#6 1
#7 0
#8 0
#9 0
#10 0



